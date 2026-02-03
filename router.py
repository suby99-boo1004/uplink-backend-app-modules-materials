from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.dependencies import get_current_user_id  # auth router와 동일 스타일 사용

# =========================================================
# 자재 요청(Material Requests) Router v1.0
# - DB 기반(material_requests, material_request_items)
# - qty_used 변경 시 재고 반영은 DB 트리거가 담당(대표님이 SQL로 추가한 구조)
# - 관리자/운영자만 재고/사용량 조회/수정 가능
# =========================================================

router = APIRouter(prefix="/api/material-requests", tags=["materials"])

ROLE_ADMIN_ID = 6
ROLE_OPERATOR_ID = 7


def _get_user(db: Session, user_id: int) -> Dict[str, Any]:
    row = db.execute(
        text(
            """
            SELECT id, name, role_id
            FROM users
            WHERE id = :id AND deleted_at IS NULL
            """
        ),
        {"id": user_id},
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=401, detail="인증이 필요합니다.")
    return dict(row)


def _is_admin_or_operator(user: Dict[str, Any]) -> bool:
    try:
        rid = int(user.get("role_id")) if user.get("role_id") is not None else None
    except Exception:
        rid = None
    return rid in (ROLE_ADMIN_ID, ROLE_OPERATOR_ID)



def _resolve_product_id_by_snapshot(db: Session, name: str, spec: str) -> Optional[int]:
    n = (name or "").strip().strip('"').strip("'").strip()
    s = (spec or "").strip().strip('"').strip("'").strip()
    if not n:
        return None
    name_pat = f"%{n}%"
    spec_pat = f"%{s}%" if s else ""
    if s:
        row = db.execute(
            text(
                """
                SELECT id
                FROM products
                WHERE deleted_at IS NULL
                  AND COALESCE(name,'') ILIKE :name_pat
                  AND COALESCE(spec,'') ILIKE :spec_pat
                ORDER BY id ASC
                LIMIT 1
                """
            ),
            {"name_pat": name_pat, "spec_pat": spec_pat},
        ).mappings().first()
        if row and row.get("id") is not None:
            return int(row["id"])
    row = db.execute(
        text(
            """
            SELECT id
            FROM products
            WHERE deleted_at IS NULL
              AND COALESCE(name,'') ILIKE :name_pat
            ORDER BY id ASC
            LIMIT 1
            """
        ),
        {"name_pat": name_pat},
    ).mappings().first()
    if row and row.get("id") is not None:
        return int(row["id"])
    return None


def require_admin_or_operator(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    user = _get_user(db, user_id)
    if _is_admin_or_operator(user):
        return user
    raise HTTPException(status_code=403, detail="관리자/운영자만 가능합니다.")


class MRItemIn(BaseModel):
    product_id: Optional[int] = None
    estimate_item_id: Optional[int] = None
    item_name_snapshot: str = Field(default="")
    spec_snapshot: str = Field(default="")
    unit_snapshot: str = Field(default="")
    qty_requested: float = Field(default=0)
    note: str = Field(default="")
    source: str = Field(default="MANUAL")  # DB enum과 다르면 서버에서 그대로 넣지 말고 MANUAL로만 사용


class MRCreateIn(BaseModel):
    project_id: Optional[int] = None
    client_id: Optional[int] = None
    estimate_id: Optional[int] = None
    estimate_revision_id: Optional[int] = None

    # 대표님 요구: 자재요청사업명(견적서 선택 or 수동 입력)
    project_name: str = Field(default="")

    warehouse_id: Optional[int] = None
    memo: str = Field(default="")
    items: List[MRItemIn] = Field(default_factory=list)


class MRUpdateIn(BaseModel):
    project_name: Optional[str] = None
    warehouse_id: Optional[int] = None
    memo: Optional[str] = None


@router.get("")
def list_material_requests(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    year: int = Query(default=0, description="0이면 서버에서 현재년도 취급(프론트 기본값)"),
    state: str = Query(default="ONGOING", description="ONGOING/DONE/CANCELED (2차에서 프로젝트 JOIN으로 고정)"),
    q: str = Query(default="", description="사업명/등록자 검색"),
):
    """
    1차: material_requests + users(등록자) 기반 리스트.
    2차: 프로젝트 business_state JOIN 후, 진행/완료/취소를 정확히 분기.
    """
    user = _get_user(db, user_id)
    can_see_sensitive = _is_admin_or_operator(user)

    # year=0이면 그냥 필터 없이 반환(프론트가 기본값으로 현재년 보내면 그때 필터)
    year_filter = ""
    params: Dict[str, Any] = {"q": f"%{q}%", "state": state.upper()}

    if year and year > 0:
        year_filter = "AND EXTRACT(YEAR FROM mr.created_at) = :year"
        params["year"] = year

    # state는 2차에서 프로젝트 JOIN으로 강제할 예정이라 1차에서는 무시(호환용)
    rows = db.execute(
        text(
            f"""
            SELECT
              mr.id,
              mr.request_no,
              mr.project_id,
              mr.estimate_id,
              COALESCE(mr.memo, '') AS memo,
              mr.status,
              mr.warehouse_id,
              mr.requested_by,
              u.name AS requested_by_name,
              p.name AS project_name,
              e.title AS estimate_title,
              COALESCE(NULLIF(p.name,''), NULLIF(mr.memo,''), NULLIF(e.title,''), NULLIF(mr.request_no,''), CONCAT('자재요청#', mr.id)) AS business_name,
              (COALESCE(e.business_state, 'ONGOING'::estimate_business_state))::text AS business_state,
              mr.created_at,
              mr.updated_at,
              -- 준비상태(요약): 1개라도 PREPARING 있으면 PREPARING
              CASE
                WHEN EXISTS (
                  SELECT 1
                  FROM material_request_items mri
                  WHERE mri.material_request_id = mr.id
                    AND mri.prep_status = 'PREPARING'
                ) THEN 'PREPARING'
                ELSE 'READY'
              END AS prep_status
            FROM material_requests mr
            LEFT JOIN users u ON u.id = mr.requested_by
            LEFT JOIN projects p ON p.id = mr.project_id
            LEFT JOIN estimates e ON e.id = mr.estimate_id
            WHERE 1=1
              {year_filter}

              AND (
                (:state = 'DONE' AND COALESCE(e.business_state, 'ONGOING'::estimate_business_state) = 'DONE'::estimate_business_state) OR
                (:state = 'CANCELED' AND COALESCE(e.business_state, 'ONGOING'::estimate_business_state) = 'CANCELED'::estimate_business_state) OR
                (:state NOT IN ('DONE','CANCELED') AND COALESCE(e.business_state, 'ONGOING'::estimate_business_state) NOT IN ('DONE'::estimate_business_state,'CANCELED'::estimate_business_state))
              )
              AND (
                COALESCE(:q, '') = '' OR
                COALESCE(u.name,'') ILIKE :q OR
                COALESCE(mr.memo,'') ILIKE :q OR
                COALESCE(p.name,'') ILIKE :q OR
                COALESCE(e.title,'') ILIKE :q OR
                COALESCE(mr.request_no,'') ILIKE :q
              )
            ORDER BY mr.id DESC
            """
        ),
        params,
    ).mappings().all()

    # 민감정보는 리스트에선 굳이 내려주지 않음(상세에서 처리)
    return {
        "can_see_sensitive": can_see_sensitive,
        "items": [dict(r) for r in rows],
    }


def _normalize_item_source(v: Any) -> str:
    """프론트(source) 값을 DB enum(mr_source) 문자열로 정규화"""
    s = (str(v) if v is not None else "").strip().upper()
    if "FROM_ESTIMATE" in s or "ESTIMATE" in s:
        return "FROM_ESTIMATE"
    if "FROM_PRODUCT" in s or "UPLINK" in s or "PRODUCT" in s:
        return "FROM_PRODUCT"
    if "FROM_DESIGN" in s or "DESIGN" in s:
        return "FROM_DESIGN"
    if "MANUAL" in s or "TEXT" in s:
        return "MANUAL_TEXT"
    return "MANUAL_TEXT"

def _decide_item_source(it: Any, header_estimate_id: Optional[int]) -> str:
    """아이템 source를 안전하게 결정.
    - estimate_item_id가 있으면 FROM_ESTIMATE
    - product_id가 있으면 FROM_PRODUCT (FK로 NULL이 되더라도 분류 목적 유지)
    - 그 외는 프론트 source 기반 정규화
    - 단, 헤더에 estimate_id가 있고 프론트 source가 비어있으면 기본 FROM_ESTIMATE로 간주(견적 기반 생성 호환)
    """
    est_item_id = getattr(it, "estimate_item_id", None)
    try:
        est_item_id = int(est_item_id) if est_item_id is not None else None
    except Exception:
        est_item_id = None

    prod_id = getattr(it, "product_id", None)
    try:
        prod_id = int(prod_id) if prod_id is not None else None
    except Exception:
        prod_id = None

    if est_item_id and est_item_id > 0:
        return "FROM_ESTIMATE"
    if prod_id and prod_id > 0:
        return "FROM_PRODUCT"

    raw = getattr(it, "source", None)
    normalized = _normalize_item_source(raw)
    if header_estimate_id and normalized == "MANUAL_TEXT":
        # 견적 기반 헤더인데 아이템 source가 비어있으면 견적 항목으로 간주
        return "FROM_ESTIMATE"
    return normalized


@router.post("")

def create_material_request(
    payload: MRCreateIn,
    v: int = Query(default=1, description="호환용 버전 파라미터"),
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(require_admin_or_operator),  # 생성은 관리자/운영자만 (정책 원하면 변경 가능)
):
    # 헤더 생성
    mr = db.execute(
        text(
            """
            WITH new_id AS (
              SELECT nextval('public.material_requests_id_seq') AS id
            )
            INSERT INTO material_requests (
              id,
              request_no,
              project_id, client_id, estimate_id, estimate_revision_id,
              status, requested_by, memo, warehouse_id
            )
            SELECT
              new_id.id,
              'MR-' || to_char(NOW(), 'YYYYMMDD') || '-' || lpad(new_id.id::text, 4, '0'),
              :project_id, :client_id, :estimate_id, :estimate_revision_id,
              'DRAFT', :requested_by, :memo, :warehouse_id
            FROM new_id
            RETURNING id
            """
        ),
        {
            "project_id": payload.project_id,
            "client_id": payload.client_id,
            "estimate_id": payload.estimate_id,
            "estimate_revision_id": payload.estimate_revision_id,
            "requested_by": int(user["id"]),
            "memo": payload.memo,
            "warehouse_id": payload.warehouse_id,
        },
    ).mappings().first()

    if not mr:
        raise HTTPException(status_code=500, detail="자재요청 생성에 실패했습니다.")

    mr_id = int(mr["id"])

    # 아이템 삽입
    for it in payload.items:
        db.execute(
            text(
                """
                INSERT INTO material_request_items (
                  material_request_id,
                  product_id,
                  estimate_item_id,
                  source,
                  item_name_snapshot,
                  spec_snapshot,
                  unit_snapshot,
                  qty_requested,
                  note,
                  prep_status,
                  qty_used
                )
                VALUES (
                  :material_request_id,
                  (SELECT id FROM products WHERE id = :product_id),
                  :estimate_item_id,
                  :source,
                  :item_name_snapshot,
                  :spec_snapshot,
                  :unit_snapshot,
                  :qty_requested,
                  :note,
                  'PREPARING',
                  0
                )
                """
            ),
            {
                "material_request_id": mr_id,
                "product_id": it.product_id,
                "estimate_item_id": it.estimate_item_id,
                "source": _decide_item_source(it, getattr(payload, "estimate_id", None)),
                "item_name_snapshot": it.item_name_snapshot,
                "spec_snapshot": it.spec_snapshot,
                "unit_snapshot": it.unit_snapshot,
                "qty_requested": it.qty_requested,
                "note": it.note,
            },
        )

    db.commit()
    row = db.execute(text("SELECT id, request_no FROM material_requests WHERE id = :id"), {"id": mr_id}).mappings().first()
    return {"id": mr_id, "request_no": (row.get("request_no") if row else None), "status": "DRAFT"}


@router.get("/{mr_id}")
def get_material_request_detail(
    mr_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    user = _get_user(db, user_id)
    can_see_sensitive = _is_admin_or_operator(user)

    header = db.execute(
        text(
            """
            SELECT
              mr.id,
              mr.request_no,
              mr.project_id,
              mr.estimate_id,
              mr.warehouse_id,
              mr.status,
              mr.requested_by,
              u.name AS requested_by_name,
              p.name AS project_name,
              e.title AS estimate_title,
              COALESCE(mr.memo,'') AS memo,
              mr.created_at,
              mr.updated_at
            FROM material_requests mr
            LEFT JOIN users u ON u.id = mr.requested_by
            LEFT JOIN projects p ON p.id = mr.project_id
            LEFT JOIN estimates e ON e.id = mr.estimate_id
            WHERE mr.id = :mr_id
            """
        ),
        {"mr_id": mr_id},
    ).mappings().first()

    if not header:
        raise HTTPException(status_code=404, detail="자재요청을 찾을 수 없습니다.")

    items = db.execute(
        text(
            """
            SELECT
              mri.id,
              mri.material_request_id,
              mri.product_id,
              mri.estimate_item_id,
              mri.source,
              COALESCE(mri.item_name_snapshot,'') AS item_name_snapshot,
              COALESCE(mri.spec_snapshot,'') AS spec_snapshot,
              COALESCE(mri.unit_snapshot,'') AS unit_snapshot,
              COALESCE(mri.qty_requested,0) AS qty_requested,
              COALESCE(mri.qty_used,0) AS qty_used,
              COALESCE(mri.note,'') AS note,
              mri.prep_status
            FROM material_request_items mri
            WHERE mri.material_request_id = :mr_id
            ORDER BY mri.id ASC
            """
        ),
        {"mr_id": mr_id},
    ).mappings().all()

    items_out: List[Dict[str, Any]] = []
    wh: Optional[int] = int(header["warehouse_id"]) if header.get("warehouse_id") else None

    for r in items:
        d = dict(r)
        qty_on_hand: Optional[float] = None

        pid: Optional[int] = int(d["product_id"]) if d.get("product_id") else None

        if pid is None and d.get("estimate_item_id"):
            est = db.execute(
                text(
                    """
                    SELECT product_id
                    FROM estimate_items
                    WHERE id = :eid
                    """
                ),
                {"eid": int(d["estimate_item_id"])},
            ).mappings().first()
            if est and est.get("product_id"):
                pid = int(est["product_id"])

        if pid is None:
            pid = _resolve_product_id_by_snapshot(db, d.get("item_name_snapshot", ""), d.get("spec_snapshot", ""))

        if pid is not None:
            if wh is not None:
                inv = db.execute(
                    text(
                        """
                        SELECT qty_on_hand
                        FROM inventory
                        WHERE warehouse_id = :wh AND product_id = :pid
                        """
                    ),
                    {"wh": wh, "pid": pid},
                ).mappings().first()
                qty_on_hand = float(inv["qty_on_hand"]) if inv and inv.get("qty_on_hand") is not None else 0.0
            else:
                inv = db.execute(
                    text(
                        """
                        SELECT COALESCE(SUM(qty_on_hand), 0) AS qty_on_hand
                        FROM inventory
                        WHERE product_id = :pid
                        """
                    ),
                    {"pid": pid},
                ).mappings().first()
                qty_on_hand = float(inv["qty_on_hand"]) if inv and inv.get("qty_on_hand") is not None else 0.0

        d["qty_on_hand"] = qty_on_hand

        if not can_see_sensitive:
            d["qty_used"] = None

        items_out.append(d)

    prep_status = "READY"
    for r in items:
        if r.get("prep_status") == "PREPARING":
            prep_status = "PREPARING"
            break

    return {
        "can_see_sensitive": can_see_sensitive,
        "header": dict(header),
        "prep_status": prep_status,
        "items": items_out,
    }



@router.put("/{mr_id}")
def update_material_request(
    mr_id: int,
    payload: MRUpdateIn,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(require_admin_or_operator),
):
    db.execute(
        text(
            """
            UPDATE material_requests
            SET
              memo = COALESCE(:memo, memo),
              warehouse_id = COALESCE(:warehouse_id, warehouse_id),
              updated_at = NOW()
            WHERE id = :id
            """
        ),
        {"id": mr_id, "memo": payload.memo, "warehouse_id": payload.warehouse_id},
    )
    db.commit()
    return {"ok": True}


@router.post("/{mr_id}/mark-all-ready")
def mark_all_ready(
    mr_id: int,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(require_admin_or_operator),
):
    # 대표님 SQL에서 만든 함수 호출 (없으면 여기서 직접 UPDATE로 바꿔도 됨)
    db.execute(text("SELECT public.fn_material_request_mark_all_ready(:id)"), {"id": mr_id})
    db.commit()
    return {"ok": True}


class MRItemPatchIn(BaseModel):
    prep_status: Optional[str] = None  # 'PREPARING'|'READY'
    qty_used: Optional[float] = None   # 관리자/운영자만
    note: Optional[str] = None
    qty_requested: Optional[float] = None


@router.patch("/items/{item_id}")
def patch_material_request_item(
    item_id: int,
    payload: MRItemPatchIn,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    user = _get_user(db, user_id)
    can_edit_sensitive = _is_admin_or_operator(user)

    # qty_used는 관리자/운영자만 수정
    if payload.qty_used is not None and not can_edit_sensitive:
        raise HTTPException(status_code=403, detail="사용량은 관리자/운영자만 수정할 수 있습니다.")

    # prep_status 값 제한
    if payload.prep_status is not None and payload.prep_status not in ("PREPARING", "READY"):
        raise HTTPException(status_code=400, detail="prep_status는 PREPARING 또는 READY 입니다.")

    # 동적 업데이트
    db.execute(
        text(
            """
            UPDATE material_request_items
            SET
              prep_status = COALESCE(:prep_status, prep_status),
              note = COALESCE(:note, note),
              qty_requested = COALESCE(:qty_requested, qty_requested),
              qty_used = COALESCE(:qty_used, qty_used)
            WHERE id = :id
            """
        ),
        {
            "id": item_id,
            "prep_status": payload.prep_status,
            "note": payload.note,
            "qty_requested": payload.qty_requested,
            "qty_used": payload.qty_used,
        },
    )

    # 준비완료(READY)로 바뀌는 순간: 재고변경수량(현재수량-사용수량)을 inventory.qty_on_hand로 확정 반영
    # - 대표님 요구사항: 준비완료 클릭 시 재고변경수량이 제품(자재관리) 수량으로 저장
    if payload.prep_status == "READY":
        row = db.execute(
            text(
                """
                SELECT
                  mri.product_id,
                  mri.qty_used,
                  mr.warehouse_id
                FROM material_request_items mri
                JOIN material_requests mr ON mr.id = mri.material_request_id
                WHERE mri.id = :id
                """
            ),
            {"id": item_id},
        ).mappings().first()
    
        if row and row.get("product_id") and row.get("warehouse_id"):
            wh = int(row["warehouse_id"])
            pid = int(row["product_id"])
            used = float(row.get("qty_used") or 0)
    
            inv = db.execute(
                text(
                    """
                    SELECT qty_on_hand
                    FROM inventory
                    WHERE warehouse_id = :wh AND product_id = :pid
                    """
                ),
                {"wh": wh, "pid": pid},
            ).mappings().first()
    
            current_qty = float(inv["qty_on_hand"]) if inv and inv.get("qty_on_hand") is not None else 0.0
            new_qty = current_qty - used
    
            # UPSERT (warehouse_id, product_id) 기준
            db.execute(
                text(
                    """
                    INSERT INTO inventory (warehouse_id, product_id, qty_on_hand)
                    VALUES (:wh, :pid, :qty)
                    ON CONFLICT (warehouse_id, product_id)
                    DO UPDATE SET qty_on_hand = EXCLUDED.qty_on_hand
                    """
                ),
                {"wh": wh, "pid": pid, "qty": new_qty},
            )

    # qty_used가 바뀌면 DB 트리거가 inventory/stock_movements 반영
    db.commit()
    return {"ok": True}