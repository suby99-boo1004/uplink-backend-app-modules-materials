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


def _pick_mr_title_column(db: Session) -> Optional[str]:
    """자재요청 '건명(사업명)' 저장 컬럼을 환경에 맞춰 선택한다.
    우선순위: business_name > project_name > title
    """
    for col in ("business_name", "project_name", "title"):
        try:
            if _has_column(db, "material_requests", col):
                return col
        except Exception:
            continue
    return None

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
    title_col = _pick_mr_title_column(db)
    title_col = _pick_mr_title_column(db)
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
    is_pinned: Optional[bool] = None


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
    has_pinned = _has_column(db, "material_requests", "is_pinned")

    prep_status_expr = """CASE
                        WHEN SUM(CASE WHEN COALESCE(mri.prep_status,'PREPARING') = 'READY' THEN 0 ELSE 1 END) = 0
                        THEN 'READY'
                        ELSE 'PREPARING'
                    END"""
    if has_overall_status:
        prep_status_expr = f"COALESCE(mr.prep_overall_status::text, {prep_status_expr})"

    pinned_expr = "COALESCE(mr.is_pinned, false)" if has_pinned else "false"

    title_col = _pick_mr_title_column(db)
    if title_col:
        title_expr = f"COALESCE(NULLIF(mr.{title_col},''), NULLIF(pj.name,''), NULLIF(est.title,''), NULLIF(mr.memo,''), ('자재요청#' || mr.id::text))"
    else:
        title_expr = "COALESCE(NULLIF(pj.name,''), NULLIF(est.title,''), NULLIF(mr.memo,''), ('자재요청#' || mr.id::text))"

    sql_list = f"""
                SELECT
                    mr.id,
                    mr.project_id,
                    mr.estimate_id,
                    mr.memo,
                    mr.created_at,
                    mr.status,
                    {pinned_expr} AS is_pinned,
                    u.name AS requested_by_name,
                    {('mr.is_pinned,' if has_pinned else 'false AS is_pinned,')}
                    -- 표기용: business_name 우선순위
                    {title_expr} AS business_name,
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
    has_pinned = _has_column(db, "material_requests", "is_pinned")

    title_col = _pick_mr_title_column(db)
    if title_col:
        title_expr = f"COALESCE(NULLIF(mr.{title_col},''), NULLIF(pj.name,''), NULLIF(est.title,''), NULLIF(mr.memo,''), ('자재요청#' || mr.id::text))"
    else:
        title_expr = "COALESCE(NULLIF(pj.name,''), NULLIF(est.title,''), NULLIF(mr.memo,''), ('자재요청#' || mr.id::text))"

    header_sql = f"""
                SELECT
                    mr.id,
            mr.project_id,
            mr.estimate_id,
                    mr.memo,
                    mr.status,
                    mr.warehouse_id,
                    mr.created_at,
                    u.name AS requested_by_name,
                    {('mr.is_pinned,' if _has_column(db, "material_requests", "is_pinned") else 'false AS is_pinned,')}
                    {('mr.prep_overall_status,' if has_overall_status else '')}
                    {('mr.prep_completed_at,' if has_overall_status else '')}
                    {title_expr} AS business_name
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





@router.post("/{mr_id}/complete")
def complete_manual_material_request(
    mr_id: int,
    body: Dict[str, Any],
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """수동 자재요청(프로젝트/견적서 미연동) 건을 사업완료로 전환한다."""
    _ = _get_user(db, user_id)  # auth

    row = (
        db.execute(
            text(
                """
                SELECT id, project_id, estimate_id, status
                FROM material_requests
                WHERE id = :id AND deleted_at IS NULL
                """
            ),
            {"id": mr_id},
        )
        .mappings()
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="자재요청을 찾을 수 없습니다.")

    # 프로젝트/견적서 연동 건은 여기서 사업완료 처리하지 않음
    if row.get("project_id") is not None or row.get("estimate_id") is not None:
        raise HTTPException(status_code=409, detail="견적서/프로젝트 연동 자재요청은 이 경로로 사업완료할 수 없습니다.")

    if not body or not bool(body.get("confirm", False)):
        raise HTTPException(status_code=400, detail="confirm 값이 필요합니다.")

    mr_status_labels = _get_enum_labels(db, "mr_status")
    done_value = _pick_mr_status_label(mr_status_labels, "DONE") if mr_status_labels else "DONE"

    has_overall_status = _has_column(db, "material_requests", "prep_overall_status")
    sets = ["status = :st", "updated_at = NOW()"]
    params = {"id": mr_id, "st": done_value}

    # 환경에 따라 존재할 수 있는 컬럼은 있으면 함께 업데이트
    if has_overall_status:
        sets.append("prep_overall_status = 'READY'")
        if _has_column(db, "material_requests", "prep_completed_at"):
            sets.append("prep_completed_at = NOW()")

    db.execute(
        text(
            f"""
            UPDATE material_requests
            SET {", ".join(sets)}
            WHERE id = :id AND deleted_at IS NULL
            """
        ),
        params,
    )
    db.commit()
    return {"ok": True, "id": mr_id, "status": done_value}

@router.patch("/{mr_id}")
def update_material_request(
    mr_id: int,
    body: MRUpdateIn,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> Dict[str, Any]:
    user = _get_user(db, user_id)  # auth

    # 상단 고정(is_pinned) 변경은 관리자/운영자만
    if body.is_pinned is not None and not _is_admin_or_operator(user):
        raise HTTPException(status_code=403, detail="관리자/운영자만 가능합니다.")

    # 최소 변경: memo/warehouse_id/is_pinned만 업데이트
    has_pinned = _has_column(db, "material_requests", "is_pinned")
    params: Dict[str, Any] = {"id": mr_id}
    sets = []
    if body.warehouse_id is not None:
        sets.append("warehouse_id = :warehouse_id")
        params["warehouse_id"] = body.warehouse_id
    if body.memo is not None:
        sets.append("memo = :memo")
        params["memo"] = (body.memo or "").strip()
    if body.is_pinned is not None and has_pinned:
        sets.append("is_pinned = :is_pinned")
        params["is_pinned"] = bool(body.is_pinned)

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
    has_pinned = _has_column(db, "material_requests", "is_pinned")
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
            {title_expr} AS business_name
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
    # ✅ '자재요청 참고사항(memo)'은 신규등록 시 기본값이 비어있어야 하므로,
    #    건명은 project_name으로만 받고 memo는 사용자가 입력할 때만 저장한다.
    business_name = (body.project_name or "").strip()
    if not business_name:
        raise HTTPException(status_code=400, detail="자재요청 참고사항을 넣으세요")

    body.project_name = business_name
    body.memo = (body.memo or "").strip()

    memo_for_db = (body.memo or "").strip()
    title_col = _pick_mr_title_column(db)
    if not title_col:
        # title 컬럼이 없는 환경: 리스트/상세 표기(business_name)를 위해 memo에 건명을 저장(참고사항은 뒤에 이어붙임)
        if memo_for_db:
            memo_for_db = f"[건명] {business_name}\n{memo_for_db}"
        else:
            memo_for_db = business_name



    # 상태는 프로젝트 탭과 동일하게 신규는 진행중
    # INSERT 컬럼을 환경에 맞춰 구성(건명 컬럼이 있으면 저장)
    mr_cols = [
        "request_no",
        "project_id",
        "client_id",
        "estimate_id",
        "estimate_revision_id",
        "warehouse_id",
        "memo",
    ]
    mr_vals = [
        "('MR-' || to_char(NOW(), 'YYYYMMDDHH24MISS') || '-' || lpad(((random()*10000)::int)::text, 4, '0'))",
        ":project_id",
        ":client_id",
        ":estimate_id",
        ":estimate_revision_id",
        ":warehouse_id",
        ":memo",
    ]
    params_insert_mr_base = {
        "project_id": body.project_id,
        "client_id": body.client_id,
        "estimate_id": body.estimate_id,
        "estimate_revision_id": body.estimate_revision_id,
        "warehouse_id": body.warehouse_id,
        "memo": memo_for_db,
        "requested_by": user["id"],
    }

    if title_col:
        mr_cols.insert(-1, title_col)
        mr_vals.insert(-1, f":{title_col}")
        params_insert_mr_base[title_col] = business_name
    # 상태/문서번호/필수값은 DB 스키마에 맞춰 안전하게 INSERT
    if 'mr_status_labels' in locals() and mr_status_labels:
        sql_insert_mr = f"""
            INSERT INTO material_requests
                ({', '.join(mr_cols + ['status', 'requested_by'])})
            VALUES
                ({', '.join(mr_vals + [':status', ':requested_by'])})
            RETURNING id
        """
        params_insert_mr = dict(params_insert_mr_base)
        params_insert_mr['status'] = status_value
    else:
        # enum 정보를 못 읽는 환경: status는 DB DEFAULT에 맡김
        sql_insert_mr = f"""
            INSERT INTO material_requests
                ({', '.join(mr_cols + ['requested_by'])})
            VALUES
                ({', '.join(mr_vals + [':requested_by'])})
            RETURNING id
        """
        params_insert_mr = dict(params_insert_mr_base)


