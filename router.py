from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.dependencies import get_current_user_id

# =========================================================
# 자재 요청(Material Requests) Router v1.1 (replace-safe)
# - DB 기반(material_requests, material_request_items, products, users, projects/estimates optional)
# - qty_used 변경 시 재고 반영은 DB 트리거/프로시저가 담당(대표님 SQL 구조)
# - 관리자/운영자만 재고/사용량 등 민감정보(can_see_sensitive) 조회/수정
# - 프론트 호환: source 값은 ESTIMATE/PRODUCT/MANUAL 또는 UPLINK_PRODUCT/MANUAL_TEXT 등도 수용
# =========================================================

router = APIRouter(prefix="/api/material-requests", tags=["materials"])

ROLE_ADMIN_ID = 6
ROLE_OPERATOR_ID = 7




def _ensure_default_warehouse_id(db: Session) -> int:
    """warehouse_id가 NULL일 때 사용할 '기준 창고'를 확보한다.
    - warehouses 테이블에 1개라도 있으면 가장 작은 id를 사용
    - 없으면 '기준창고'를 생성 후 id 반환
    """
    try:
        row = db.execute(text("SELECT id FROM warehouses ORDER BY id ASC LIMIT 1")).mappings().first()
        if row and row.get("id") is not None:
            return int(row["id"])
        row2 = db.execute(
            text("INSERT INTO warehouses (name) VALUES (:name) RETURNING id"),
            {"name": "기준창고"},
        ).mappings().first()
        db.flush()
        return int(row2["id"])
    except Exception as e:
        # warehouses 테이블이 없거나 권한/스키마 문제
        raise HTTPException(
            status_code=409,
            detail=f"창고(warehouses) 테이블을 확인해주세요. 기본창고를 확보할 수 없습니다. 원인: {e}",
        )
# pg enum label cache (process-local)
_ENUM_LABELS_CACHE: Dict[str, List[str]] = {}



def _pick_mr_status_label(mr_status_labels: List[str], desired: str) -> Optional[str]:
    """프론트 state(ONGOING/DONE/CANCELED)를 DB enum(mr_status) 실제 값으로 매핑
    - 매핑 실패 시 잘못된 값(예: DONE으로 오인)으로 떨어지지 않도록, 원본 desired를 그대로 쓰도록 한다.
    """
    desired_up = (desired or "").strip().upper()
    if not mr_status_labels:
        return desired_up or None

    labels_up = [x.upper() for x in (mr_status_labels or [])]

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            cu = (c or "").upper()
            if cu in labels_up:
                idx = labels_up.index(cu)
                return mr_status_labels[idx]
        return None

    if desired_up == "DONE":
        # DONE 계열이 없으면 desired를 그대로 반환(결과 0건이 더 안전)
        return pick(["DONE", "COMPLETE", "COMPLETED", "FINISHED", "CLOSED", "END"]) or desired_up

    if desired_up == "CANCELED":
        # CANCELED 계열이 없으면 desired를 그대로 반환(결과 0건이 더 안전)
        return pick(["CANCELED", "CANCELLED", "CANCEL", "ABORT", "ABORTED", "VOID"]) or desired_up

    # ONGOING
    return pick(["ONGOING", "IN_PROGRESS", "PROGRESS", "ACTIVE", "RUNNING", "OPEN", "DRAFT", "NEW"]) or desired_up


def _get_enum_labels(db: Session, type_name: str) -> List[str]:
    """PostgreSQL enum 라벨 목록을 조회합니다. (예: mr_source, mr_status)
    - DB 스키마가 환경마다 다르므로, 런타임에 실제 enum 값에 맞춰 매핑하기 위함
    """
    if type_name in _ENUM_LABELS_CACHE:
        return _ENUM_LABELS_CACHE[type_name]

    rows = (
        db.execute(
            text(
                """
                SELECT e.enumlabel::text AS label
                FROM pg_type t
                JOIN pg_enum e ON t.oid = e.enumtypid
                WHERE t.typname = :typ
                ORDER BY e.enumsortorder
                """
            ),
            {"typ": type_name},
        )
        .mappings()
        .all()
    )
    labels = [r["label"] for r in rows] if rows else []
    _ENUM_LABELS_CACHE[type_name] = labels
    return labels

def _get_products_qty_expr(db: Session) -> str:
    """products 테이블의 '현재수량/재고' 컬럼명을 환경별로 흡수해서 qty_on_hand 표현식을 만든다.

    - 존재하는 컬럼만 사용
    - 없으면 0을 반환
    """
    try:
        rows = (
            db.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = 'products'
                    """
                )
            )
            .mappings()
            .all()
        )
        cols_all = [r["column_name"] for r in rows]
    except Exception:
        cols_all = []

    candidates = [
        "qty_on_hand",
        "on_hand",
        "stock_qty",
        "current_qty",
        "inventory_qty",
        "qty",
        "quantity",
    ]

    cols = [c for c in candidates if c in cols_all]
    if not cols:
        return "0"

    # 안전하게 double-quote로 감싸기(대소문자/특수문자 대응)
    expr_parts = [f'p."{c}"' for c in cols]
    return f"COALESCE({', '.join(expr_parts)}, 0)"


def _has_column(db: Session, table: str, column: str, schema: str = "public") -> bool:
    row = (
        db.execute(
            text(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND table_name = :table
                  AND column_name = :column
                LIMIT 1
                """
            ),
            {"schema": schema, "table": table, "column": column},
        )
        .mappings()
        .first()
    )
    return bool(row)

def _pick_mr_source_label(mr_source_labels: List[str], logical: str) -> str:
    """논리적 source(ESTIMATE/PRODUCT/MANUAL)를 DB enum(mr_source) 실제 라벨로 안전 매핑
    - PRODUCT는 절대 ESTIMATE 계열로 매핑되지 않도록 방지
    """
    labels = [x.upper() for x in (mr_source_labels or [])]
    if not labels:
        return logical

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            cu = c.upper()
            if cu in labels:
                i = labels.index(cu)
                return mr_source_labels[i]
        return None

    # 라벨 그룹(대/소문자 변형 흡수)
    est_like = ["ESTIMATE", "EST", "QUOTE", "ESTIMATION", "FROM_ESTIMATE", "FROM_QUOTE"]
    prod_like = ["PRODUCT", "UPLINK_PRODUCT", "ITEM", "GOODS", "MATERIAL", "STOCK"]
    man_like  = ["MANUAL", "DIRECT", "CUSTOM", "ETC", "TEXT", "FREE"]

    logical = (logical or "").strip().upper() or "MANUAL"

    if logical == "ESTIMATE":
        chosen = pick(est_like) or pick(["FROM_ESTIMATE"])  # 우선 견적 계열
        return chosen or mr_source_labels[0]

    if logical == "PRODUCT":
        chosen = pick(prod_like)
        if chosen:
            return chosen
        # PRODUCT인데 prod_like 매칭이 안 되면: ESTIMATE 계열을 피해서 첫 비-견적 라벨 선택
        for i, lab_up in enumerate(labels):
            if lab_up not in [x.upper() for x in est_like]:
                return mr_source_labels[i]
        return mr_source_labels[0]

    # MANUAL
    chosen = pick(man_like)
    if chosen:
        return chosen
    # MANUAL도 견적계열만 피해서 선택
    for i, lab_up in enumerate(labels):
        if lab_up not in [x.upper() for x in est_like]:
            return mr_source_labels[i]
    return mr_source_labels[0]


def _get_user(db: Session, user_id: int) -> Dict[str, Any]:
    row = (
        db.execute(
            text(
                """
                SELECT id, name, role_id
                FROM users
                WHERE id = :id AND deleted_at IS NULL
                """
            ),
            {"id": user_id},
        )
        .mappings()
        .first()
    )
    if not row:
        raise HTTPException(status_code=401, detail="인증이 필요합니다.")
    return dict(row)


def _is_admin_or_operator(user: Dict[str, Any]) -> bool:
    try:
        rid = int(user.get("role_id")) if user.get("role_id") is not None else None
    except Exception:
        rid = None
    return rid in (ROLE_ADMIN_ID, ROLE_OPERATOR_ID)


def require_admin_or_operator(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    user = _get_user(db, user_id)

    # enum labels (DB 스키마 자동 적응)
    mr_source_labels = _get_enum_labels(db, 'mr_source')
    prep_status_labels = _get_enum_labels(db, 'mr_item_prep_status')
    mr_status_labels = _get_enum_labels(db, 'mr_status')
    status_value = _pick_mr_status_label(mr_status_labels, 'ONGOING') if mr_status_labels else None
    if _is_admin_or_operator(user):
        return user
    raise HTTPException(status_code=403, detail="관리자/운영자만 가능합니다.")


def _normalize_source(src: Optional[str], mr_source_labels: Optional[List[str]] = None) -> str:
    """프론트에서 들어오는 source 값을 DB enum(mr_source) 실제 값에 맞춰 안전하게 변환
    - PRODUCT는 절대 ESTIMATE 계열로 저장되지 않도록 방지
    """
    s = (src or "").strip().upper()

    if not s:
        logical = "MANUAL"
    elif "ESTIMATE" in s or "FROM_ESTIMATE" in s or "QUOTE" in s:
        logical = "ESTIMATE"
    elif "PRODUCT" in s or "UPLINK" in s or "MATERIAL" in s or "STOCK" in s or "ITEM" in s or "GOODS" in s:
        logical = "PRODUCT"
    elif "MANUAL" in s or "TEXT" in s or "DIRECT" in s or "CUSTOM" in s:
        logical = "MANUAL"
    else:
        logical = "MANUAL"

    labels = mr_source_labels or []
    return _pick_mr_source_label(labels, logical)


class MRItemIn(BaseModel):
    product_id: Optional[int] = None
    estimate_item_id: Optional[int] = None
    item_name_snapshot: str = Field(default="")
    spec_snapshot: str = Field(default="")
    unit_snapshot: str = Field(default="")
    qty_requested: float = Field(default=0)
    note: str = Field(default="")
    source: str = Field(default="MANUAL")


class MRCreateIn(BaseModel):
    project_id: Optional[int] = None
    client_id: Optional[int] = None
    estimate_id: Optional[int] = None
    estimate_revision_id: Optional[int] = None
    project_name: str = Field(default="")
    warehouse_id: Optional[int] = None
    memo: str = Field(default="")
    items: List[MRItemIn] = Field(default_factory=list)


class MRUpdateIn(BaseModel):
    warehouse_id: Optional[int] = None
    memo: Optional[str] = None


@router.get("")
def list_material_requests(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    year: int = Query(default=0, description="0이면 서버에서 필터 완화(전체/현재년도)"),
    state: str = Query(default="ONGOING", description="ONGOING/DONE/CANCELED"),
    q: str = Query(default="", description="사업명/등록자 검색"),
) -> Dict[str, Any]:
    """
    리스트:
    - 프로젝트 진행/완료/취소 탭과 동일하게 동작: state로 필터
    - 표시용 business_name: COALESCE(project_name, project.name, memo, estimate.title)
    """
    _ = _get_user(db, user_id)  # auth

    st = (state or "ONGOING").strip().upper()
    mr_status_labels = _get_enum_labels(db, 'mr_status')
    st_db = _pick_mr_status_label(mr_status_labels, st) if mr_status_labels else st
    kw = (q or "").strip()

    # year=0: 필터 완화 (프론트 신규등록에서 usedEstimateIds 체크용)
    year_filter = ""
    params: Dict[str, Any] = {"st": st_db, "kw": f"%{kw}%"}

    if year and year > 0:
        year_filter = "AND EXTRACT(YEAR FROM mr.created_at) = :yy"
        params["yy"] = year

    q_filter = ""
    if kw:
        q_filter = """
        AND (
            COALESCE(pj.name,'') ILIKE :kw
            OR COALESCE(est.title,'') ILIKE :kw
            OR COALESCE(mr.memo,'') ILIKE :kw
            OR COALESCE(u.name,'') ILIKE :kw
            OR COALESCE(pj.name,'') ILIKE :kw
            OR COALESCE(est.title,'') ILIKE :kw
        )
        """

    has_overall_status = _has_column(db, "material_requests", "prep_overall_status")

    prep_status_expr = """CASE
                        WHEN SUM(CASE WHEN COALESCE(mri.prep_status,'PREPARING') = 'READY' THEN 0 ELSE 1 END) = 0
                        THEN 'READY'
                        ELSE 'PREPARING'
                    END"""
    if has_overall_status:
        prep_status_expr = f"COALESCE(mr.prep_overall_status::text, {prep_status_expr})"

    sql_list = f"""
                SELECT
                    mr.id,
                    mr.project_id,
                    mr.estimate_id,
                    mr.memo,
                    mr.created_at,
                    mr.status,
                    u.name AS requested_by_name,
                    -- 표기용: business_name 우선순위
                    COALESCE(NULLIF(pj.name,''), NULLIF(est.title,''), NULLIF(mr.memo,''), ('자재요청#' || mr.id::text)) AS business_name,
                    -- 준비상태(READY/PREPARING/ADDITIONAL)
                    {prep_status_expr} AS prep_status
                FROM material_requests mr
                LEFT JOIN users u ON u.id = mr.requested_by AND u.deleted_at IS NULL
                LEFT JOIN projects pj ON pj.id = mr.project_id AND pj.deleted_at IS NULL
                LEFT JOIN estimates est ON est.id = mr.estimate_id
                LEFT JOIN material_request_items mri ON mri.material_request_id = mr.id AND mri.deleted_at IS NULL
                WHERE mr.deleted_at IS NULL
                  AND COALESCE(mr.status::text, :st) = :st
                  {year_filter}
                  {q_filter}
                GROUP BY mr.id, u.name, pj.name, est.title
                ORDER BY mr.created_at DESC, mr.id DESC
                """

    rows = (
        db.execute(
            text(sql_list),
            params,
        )
        .mappings()
        .all()
    )

    items = [dict(r) for r in rows]
    return {"items": items}


@router.get("/{mr_id}")
def get_material_request_detail(
    mr_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> Dict[str, Any]:
    user = _get_user(db, user_id)

    mr_source_labels = _get_enum_labels(db, 'mr_source')
    can_see_sensitive = _is_admin_or_operator(user)


    # 헤더 전체상태 컬럼(prep_overall_status)이 DB에 존재하는지(환경 호환)
    has_overall_status = _has_column(db, "material_requests", "prep_overall_status")
    header_sql = f"""
                SELECT
                    mr.id,
                    mr.memo,
                    mr.status,
                    mr.warehouse_id,
                    mr.created_at,
                    u.name AS requested_by_name,
                    {('mr.prep_overall_status,' if has_overall_status else '')}
                    {('mr.prep_completed_at,' if has_overall_status else '')}
                    COALESCE(NULLIF(pj.name,''), NULLIF(est.title,''), NULLIF(mr.memo,''), ('자재요청#' || mr.id::text)) AS business_name
                FROM material_requests mr
                LEFT JOIN users u ON u.id = mr.requested_by AND u.deleted_at IS NULL
                LEFT JOIN projects pj ON pj.id = mr.project_id AND pj.deleted_at IS NULL
                LEFT JOIN estimates est ON est.id = mr.estimate_id
                WHERE mr.id = :id AND mr.deleted_at IS NULL
                """

    header = (
        db.execute(text(header_sql), {"id": mr_id})
        .mappings()
        .first()
    )
    if not header:
        raise HTTPException(status_code=404, detail="자재요청을 찾을 수 없습니다.")

    # qty_on_hand: products 테이블 컬럼명이 환경마다 달라질 수 있어 COALESCE로 흡수
    qty_expr = _get_products_qty_expr(db)
    sql_items = f"""                SELECT
                    mri.id,
                    mri.source,
                    mri.estimate_item_id,
                    mri.product_id,
                    mri.item_name_snapshot,
                    mri.spec_snapshot,
                    mri.unit_snapshot,
                    mri.qty_requested,
                    mri.qty_used,
                    mri.prep_status,
                    mri.note,
                    mri.qty_on_hand_snapshot,
                    mri.remaining_qty_snapshot,
                    CASE
                        WHEN COALESCE(mri.prep_status,'PREPARING') = 'READY' THEN mri.qty_on_hand_snapshot
                        WHEN mri.product_id IS NULL THEN NULL
                        ELSE {qty_expr}
                    END AS qty_on_hand,
                    CASE
                        WHEN COALESCE(mri.prep_status,'PREPARING') = 'READY' THEN mri.remaining_qty_snapshot
                        WHEN mri.product_id IS NULL THEN NULL
                        ELSE ({qty_expr} - COALESCE(mri.qty_used,0))
                    END AS qty_remaining
                FROM material_request_items mri
                LEFT JOIN products p ON p.id = mri.product_id AND p.deleted_at IS NULL
                WHERE mri.material_request_id = :mr_id AND mri.deleted_at IS NULL
                ORDER BY mri.id ASC
                
"""
    rows = (
        db.execute(
            text(sql_items),
            {"mr_id": mr_id, "can": can_see_sensitive},
        )
        .mappings()
        .all()
    )

    items = [dict(r) for r in rows]

    
    # header-level prep_status(메인 표시용)
    # - DB에 prep_overall_status가 있으면 그대로 사용(PREPARING/READY/ADDITIONAL)
    # - 없으면 아이템 집계로 READY/PREPARING 계산
    prep_status = (header.get("prep_overall_status") or "").strip() if header and isinstance(header, dict) else ""
    if not prep_status:
        prep_status = "READY"
        for it in items:
            if (it.get("prep_status") or "PREPARING") != "READY":
                prep_status = "PREPARING"
                break

    return {
        "can_see_sensitive": can_see_sensitive,
        "header": dict(header),
        "prep_status": prep_status,
        "items": items,
    }




@router.patch("/{mr_id}")
def update_material_request(
    mr_id: int,
    body: MRUpdateIn,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> Dict[str, Any]:
    _ = _get_user(db, user_id)  # auth

    # 최소 변경: memo/warehouse_id만 업데이트
    params: Dict[str, Any] = {"id": mr_id}
    sets = []
    if body.warehouse_id is not None:
        sets.append("warehouse_id = :warehouse_id")
        params["warehouse_id"] = body.warehouse_id
    if body.memo is not None:
        sets.append("memo = :memo")
        params["memo"] = (body.memo or "").strip()

    if not sets:
        return {"ok": True}

    sql = f"""
        UPDATE material_requests
        SET {", ".join(sets)}, updated_at = NOW()
        WHERE id = :id AND deleted_at IS NULL
    """
    db.execute(text(sql), params)
    db.commit()

    # 최신 헤더 반환(프론트 동기화용)
    has_overall_status = _has_column(db, "material_requests", "prep_overall_status")
    header_sql = f"""
        SELECT
            mr.id,
            mr.memo,
            mr.status,
            mr.warehouse_id,
            mr.created_at,
            u.name AS requested_by_name,
            {('mr.prep_overall_status,' if has_overall_status else '')}
            {('mr.prep_completed_at,' if has_overall_status else '')}
            COALESCE(NULLIF(pj.name,''), NULLIF(est.title,''), NULLIF(mr.memo,''), ('자재요청#' || mr.id::text)) AS business_name
        FROM material_requests mr
        LEFT JOIN users u ON u.id = mr.requested_by AND u.deleted_at IS NULL
        LEFT JOIN projects pj ON pj.id = mr.project_id AND pj.deleted_at IS NULL
        LEFT JOIN estimates est ON est.id = mr.estimate_id
        WHERE mr.id = :id AND mr.deleted_at IS NULL
    """
    header = db.execute(text(header_sql), {"id": mr_id}).mappings().first()
    return {"ok": True, "header": dict(header) if header else None}
@router.post("")
def create_material_request(
    body: MRCreateIn,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> Dict[str, Any]:
    user = _get_user(db, user_id)

    mr_source_labels = _get_enum_labels(db, 'mr_source')
    # (지시) 견적서 선택 없이도 등록 가능하지만, '자재요청 건명'은 필수
    business_name = (body.memo or body.project_name or "").strip()
    if not business_name:
        raise HTTPException(status_code=400, detail="자재요청 건명을 넣으세요")
    body.memo = business_name
    body.project_name = business_name


    # 상태는 프로젝트 탭과 동일하게 신규는 진행중
    # 상태/문서번호/필수값은 DB 스키마에 맞춰 안전하게 INSERT
    if 'mr_status_labels' in locals() and mr_status_labels:
        sql_insert_mr = """
            INSERT INTO material_requests
                (request_no, project_id, client_id, estimate_id, estimate_revision_id, warehouse_id, memo, status, requested_by)
            VALUES
                (('MR-' || to_char(NOW(), 'YYYYMMDDHH24MISS') || '-' || lpad(((random()*10000)::int)::text, 4, '0')),
                 :project_id, :client_id, :estimate_id, :estimate_revision_id, :warehouse_id, :memo, :status, :requested_by)
            RETURNING id
        """
        params_insert_mr = {
            "project_id": body.project_id,
            "client_id": body.client_id,
            "estimate_id": body.estimate_id,
            "estimate_revision_id": body.estimate_revision_id,
            "warehouse_id": body.warehouse_id,
            "memo": (body.memo or "").strip(),
            "status": status_value,
            "requested_by": user["id"],
        }
    else:
        # enum 정보를 못 읽는 환경: status는 DB DEFAULT에 맡김
        sql_insert_mr = """
            INSERT INTO material_requests
                (request_no, project_id, client_id, estimate_id, estimate_revision_id, warehouse_id, memo, requested_by)
            VALUES
                (('MR-' || to_char(NOW(), 'YYYYMMDDHH24MISS') || '-' || lpad(((random()*10000)::int)::text, 4, '0')),
                 :project_id, :client_id, :estimate_id, :estimate_revision_id, :warehouse_id, :memo, :requested_by)
            RETURNING id
        """
        params_insert_mr = {
            "project_id": body.project_id,
            "client_id": body.client_id,
            "estimate_id": body.estimate_id,
            "estimate_revision_id": body.estimate_revision_id,
            "warehouse_id": body.warehouse_id,
            "memo": (body.memo or "").strip(),
            "requested_by": user["id"],
        }

    mr_row = (
        db.execute(text(sql_insert_mr), params_insert_mr)
        .mappings()
        .first()
    )
    mr_id = int(mr_row["id"])

    # 아이템 삽입
    for it in body.items or []:
        src = _normalize_source(it.source, mr_source_labels)
        # 수동 추가인데 product_id/estimate_item_id가 없으면 MANUAL로 저장(가능한 라벨로 매핑)
        if (it.product_id is None or int(it.product_id or 0) == 0) and (it.estimate_item_id is None or int(it.estimate_item_id or 0) == 0):
            src = _pick_mr_source_label(mr_source_labels, 'MANUAL')
        # 업링크 제품 추가인데 source가 견적서로 저장되는 것 방지
        if it.product_id is not None and int(it.product_id or 0) > 0 and (it.estimate_item_id is None or int(it.estimate_item_id or 0) == 0):
            src = _pick_mr_source_label(mr_source_labels, 'PRODUCT')

        name = (it.item_name_snapshot or "").strip()
        if not name:
            continue

        unit = (it.unit_snapshot or "EA").strip() or "EA"

        db.execute(
            text(
                """
                INSERT INTO material_request_items
                    (material_request_id, source, product_id, estimate_item_id, item_name_snapshot, spec_snapshot, unit_snapshot, qty_requested, qty_used, prep_status, note)
                VALUES
                    (:mr_id, :source, :product_id, :estimate_item_id, :name, :spec, :unit, :qty_requested, 0, 'PREPARING', :note)
                """
            ),
            {
                "mr_id": mr_id,
                "source": src,
                "product_id": it.product_id,
                "estimate_item_id": it.estimate_item_id,
                "name": name,
                "spec": (it.spec_snapshot or "").strip(),
                "unit": unit,
                "qty_requested": float(it.qty_requested or 0),
                "note": (it.note or "").strip(),
            },
        )

    db.commit()
    return {"id": mr_id}


@router.delete("/{mr_id}")
def delete_material_request(
    mr_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> Dict[str, Any]:
    user = _get_user(db, user_id)

    mr_source_labels = _get_enum_labels(db, 'mr_source')
    if not _is_admin_or_operator(user):
        raise HTTPException(status_code=403, detail="관리자/운영자만 가능합니다.")

    # soft delete
    db.execute(
        text(
            """
            UPDATE material_requests
            SET deleted_at = NOW()
            WHERE id = :id AND deleted_at IS NULL
            """
        ),
        {"id": mr_id},
    )
    db.execute(
        text(
            """
            UPDATE material_request_items
            SET deleted_at = NOW()
            WHERE material_request_id = :id AND deleted_at IS NULL
            """
        ),
        {"id": mr_id},
    )
    db.commit()
    return {"ok": True, "id": mr_id}


@router.post("/{mr_id}/items")
def add_material_request_item(
    mr_id: int,
    body: MRItemIn,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> Dict[str, Any]:
    _ = _get_user(db, user_id)

    # 존재 확인
    exists = (
        db.execute(
            text("SELECT id FROM material_requests WHERE id = :id AND deleted_at IS NULL"),
            {"id": mr_id},
        )
        .mappings()
        .first()
    )
    if not exists:
        raise HTTPException(status_code=404, detail="자재요청을 찾을 수 없습니다.")

    mr_source_labels = _get_enum_labels(db, 'mr_source')

    src = _normalize_source(body.source, mr_source_labels)

    # 수동 추가인데 product_id/estimate_item_id가 없으면 MANUAL로 저장(가능한 라벨로 매핑)
    if (body.product_id is None or int(body.product_id or 0) == 0) and (body.estimate_item_id is None or int(body.estimate_item_id or 0) == 0):
        src = _pick_mr_source_label(mr_source_labels, 'MANUAL')

    # 업링크 제품 추가인데 source가 견적서로 저장되는 것 방지
    if body.product_id is not None and int(body.product_id or 0) > 0 and (body.estimate_item_id is None or int(body.estimate_item_id or 0) == 0):
        src = _pick_mr_source_label(mr_source_labels, 'PRODUCT')
    name = (body.item_name_snapshot or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="자재명(item_name_snapshot)은 필수입니다.")

    unit = (body.unit_snapshot or "EA").strip() or "EA"

    row = (
        db.execute(
            text(
                """
                INSERT INTO material_request_items
                    (material_request_id, source, product_id, estimate_item_id, item_name_snapshot, spec_snapshot, unit_snapshot, qty_requested, qty_used, prep_status, note)
                VALUES
                    (:mr_id, :source, :product_id, :estimate_item_id, :name, :spec, :unit, :qty_requested, 0, 'PREPARING', :note)
                RETURNING id
                """
            ),
            {
                "mr_id": mr_id,
                "source": src,
                "product_id": body.product_id,
                "estimate_item_id": body.estimate_item_id,
                "name": name,
                "spec": (body.spec_snapshot or "").strip(),
                "unit": unit,
                "qty_requested": float(body.qty_requested or 0),
                "note": (body.note or "").strip(),
            },
        )
        .mappings()
        .first()
    )

    db.commit()
    return {"id": int(row["id"])}


@router.patch("/items/{item_id}")
def patch_material_request_item(
    item_id: int,
    body: Dict[str, Any],
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> Dict[str, Any]:
    user = _get_user(db, user_id)

    mr_source_labels = _get_enum_labels(db, 'mr_source')

    # 현재 항목 원본 값 조회(수량변경/견적항목 잠금 판단)
    cur = (
        db.execute(
            text(
                """
                SELECT id, source, qty_requested, qty_used
                FROM material_request_items
                WHERE id = :id AND deleted_at IS NULL
                """
            ),
            {"id": item_id},
        )
        .mappings()
        .first()
    )
    if not cur:
        raise HTTPException(status_code=404, detail="항목을 찾을 수 없습니다.")

    cur_source_norm = _normalize_source(cur.get("source"), mr_source_labels)
    cur_qty_requested = float(cur.get("qty_requested") or 0)
    cur_qty_used = None if cur.get("qty_used") is None else float(cur.get("qty_used") or 0)

    # qty_used는 재고와 연결되므로 관리자/운영자만
    if "qty_used" in body and not _is_admin_or_operator(user):
        raise HTTPException(status_code=403, detail="재고/사용수량 수정은 관리자/운영자만 가능합니다.")

    # (지시1) 견적서 재료비(ESTIMATE) 항목은 요청수량 변경 금지
    if "qty_requested" in body and cur_source_norm == "ESTIMATE":
        raise HTTPException(status_code=403, detail="견적서 재료비 항목의 요청수량은 변경할 수 없습니다.")

    # 허용 필드만 반영
    fields: Dict[str, Any] = {}
    prep_status_labels = _get_enum_labels(db, "mr_item_prep_status")

    # 요청수량 변경(준비중에서만 의미 있음; READY 잠금은 DB 트리거가 처리)
    if "qty_requested" in body:
        fields["qty_requested"] = float(body.get("qty_requested") or 0)

    if "qty_used" in body:
        fields["qty_used"] = float(body.get("qty_used") or 0)

    # 준비상황 직접 변경(READY / PREPARING / CHANGED)
    # 단, qty_requested 변경으로 CHANGED가 이미 설정되면 덮어쓰지 않음
    if "prep_status" in body and "prep_status" not in fields:
        ps = (body.get("prep_status") or "").strip().upper()
        if ps == "READY":
            fields["prep_status"] = "READY"
            # (READY 확정) 프론트는 prep_status만 보내므로, 사용수량이 비어있으면 요청수량을 사용수량으로 확정한다.
            # 재고 반영은 DB 트리거(fn_mri_apply_qty_used_delta)가 qty_used 변화(Δ)로 1회만 처리한다.
            if "qty_used" not in body:
                if (cur_qty_used is None or cur_qty_used == 0) and cur_qty_requested > 0:
                    fields["qty_used"] = cur_qty_requested
        elif ps == "CHANGED":
            fields["prep_status"] = "CHANGED"
        else:
            fields["prep_status"] = "PREPARING"

    if "note" in body:
        fields["note"] = (body.get("note") or "").strip()
    if not fields:
        return {"ok": True}

    # 동적 UPDATE
    set_sql = ", ".join([f"{k} = :{k}" for k in fields.keys()])
    fields["id"] = item_id

    # qty_used 변경 시 DB 트리거(fn_mri_apply_qty_used_delta)가 창고 기준을 요구하는 환경이 있습니다.
    # material_requests.warehouse_id 가 NULL이면 500이 나므로, 여기서 기본창고를 자동 세팅합니다.
    if "qty_used" in fields:
        mr = (
            db.execute(
                text(
                    "SELECT material_request_id FROM material_request_items WHERE id = :id AND deleted_at IS NULL"
                ),
                {"id": item_id},
            )
            .mappings()
            .first()
        )
        if mr and mr.get("material_request_id") is not None:
            mr_id = int(mr["material_request_id"])
            # warehouse_id가 비어있으면 기본창고 세팅
            wid = _ensure_default_warehouse_id(db)
            db.execute(
                text(
                    "UPDATE material_requests SET warehouse_id = COALESCE(warehouse_id, :wid) WHERE id = :mr_id"
                ),
                {"wid": wid, "mr_id": mr_id},
            )
            db.flush()

    try:
        row = (
            db.execute(
                text(
                    f"""
                    UPDATE material_request_items
                    SET {set_sql}
                    WHERE id = :id AND deleted_at IS NULL
                    RETURNING id
                    """
                ),
                fields,
            )
            .mappings()
            .first()
        )
    except Exception as e:
        msg = str(e)
        if "READY 상태" in msg and "수정" in msg:
            raise HTTPException(status_code=409, detail="준비완료(READY) 상태의 항목은 수정할 수 없습니다. 동일 품목 증감은 라인 추가로 처리하세요.")
        # DB enum(mr_item_prep_status)에 CHANGED가 없으면 enum 입력 오류가 발생한다.
        if "invalid input value for enum" in msg and "mr_item_prep_status" in msg and "CHANGED" in msg:
            raise HTTPException(
                status_code=409,
                detail="DB enum mr_item_prep_status에 CHANGED 값이 없습니다. 아래 SQL을 1회 실행 후 재시도하세요.\nALTER TYPE mr_item_prep_status ADD VALUE 'CHANGED';",
            )
        raise
    if not row:
        raise HTTPException(status_code=404, detail="항목을 찾을 수 없습니다.")

    db.commit()
    return {"ok": True}


@router.delete("/items/{item_id}")
def delete_material_request_item(
    item_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> Dict[str, Any]:
    user = _get_user(db, user_id)

    mr_source_labels = _get_enum_labels(db, 'mr_source')
    if not _is_admin_or_operator(user):
        raise HTTPException(status_code=403, detail="삭제는 관리자/운영자만 가능합니다.")

    row = (
        db.execute(
            text(
                """
                UPDATE material_request_items
                SET deleted_at = NOW()
                WHERE id = :id AND deleted_at IS NULL
                RETURNING id
                """
            ),
            {"id": item_id},
        )
        .mappings()
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="항목을 찾을 수 없습니다.")

    db.commit()
    return {"ok": True}
