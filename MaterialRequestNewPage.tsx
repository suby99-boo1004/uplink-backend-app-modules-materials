import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../../lib/api";
import { useAuth } from "../../lib/auth";

/**
 * 자재 요청 신규 등록
 * 이번 수정(대표님 지시, 다른 기능 고정):
 * 1) 상단 "자재요청사업명" 옆에 등록자(로그인 사용자) 표시
 * 2) 테이블 첫 컬럼(구분: 견적서 재료비/업링크 제품/수동추가) 삭제
 * 3) 컬럼 순서/명칭 변경:
 *    자재명-규격-단위-요청수량-재고수량-사용수량-준비상황-비고
 *
 * 주의:
 * - 재고수량/사용수량/준비상황은 다음 단계에서 권한/연동 로직을 붙일 예정
 * - 이번 단계는 "표시/구조"만 맞춤
 */

type EstimateRow = {
  id: number;
  title?: string | null;
  project_name?: string | null;
  client_name?: string | null;
  created_at?: string | null;
};

type Mode = "ESTIMATE" | "MANUAL";
type LineSource = "ESTIMATE" | "UPLINK_PRODUCT" | "MANUAL";
type PrepStatus = "PREPARING" | "READY";

type MaterialLine = {
  id: string;
  source: LineSource;
  product_id: number | null;

  name: string;
  spec: string;
  unit: string;

  qty: number; // 요청수량
  stock_qty: number | null; // 재고수량(표시만)
  used_qty: number | null; // 사용수량(표시만)
  prep_status: PrepStatus; // 준비상황

  note: string; // 비고
};

type ProductRow = {
  id: number;
  name: string;
  spec?: string | null;
  unit?: string | null;
};

function labelOfEstimate(e: EstimateRow) {
  const name = (e.project_name && e.project_name.trim()) || (e.title && e.title.trim()) || `견적서 #${e.id}`;
  const client = e.client_name ? ` / ${e.client_name}` : "";
  return `${name}${client}`;
}

function toNum(v: any): number | null {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  if (Number.isNaN(n)) return null;
  return n;
}

function mkId() {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function normalizeSectionType(v: any): string {
  return (v ?? "").toString().trim().toUpperCase();
}

function extractMaterialLinesFromEstimateDetail(detail: any): MaterialLine[] {
  if (!detail || typeof detail !== "object") return [];

  const sections =
    Array.isArray(detail.sections) ? detail.sections : Array.isArray(detail?.data?.sections) ? detail.data.sections : null;
  if (!sections) return [];

  const materialSections = sections.filter((s: any) => normalizeSectionType(s?.section_type) === "MATERIAL");
  if (materialSections.length === 0) return [];

  const lines: any[] = materialSections.flatMap((s: any) => (Array.isArray(s?.lines) ? s.lines : []));

  const mapped = lines
    .map((l: any) => {
      const sourceType = (l?.source_type ?? "").toString().toUpperCase();
      const productId =
        sourceType === "PRODUCT"
          ? toNum(l?.source_id)
          : toNum(l?.product_id) ?? toNum(l?.productId) ?? toNum(l?.product?.id) ?? null;

      const name = (l?.name ?? l?.item_name_snapshot ?? l?.product_name ?? "").toString().trim();
      const spec = (l?.spec ?? l?.spec_snapshot ?? "").toString().trim();
      const unit = (l?.unit ?? l?.unit_snapshot ?? "EA").toString().trim();
      const qty = toNum(l?.qty) ?? toNum(l?.qty_requested) ?? 0;

      if (!name) return null;

      return {
        id: mkId(),
        source: "ESTIMATE" as const,
        product_id: productId,
        name,
        spec,
        unit,
        qty: qty && qty > 0 ? qty : 0,
        stock_qty: null,
        used_qty: null,
        prep_status: "PREPARING" as const,
        note: "",
      };
    })
    .filter(Boolean) as MaterialLine[];

  const uniq: MaterialLine[] = [];
  const set = new Set<string>();
  for (const m of mapped) {
    const k = `${m.product_id ?? ""}|${m.name}|${m.spec}|${m.unit}|${m.source}`;
    if (set.has(k)) continue;
    set.add(k);
    uniq.push(m);
  }
  return uniq;
}

export default function MaterialRequestNewPage() {
  const navigate = useNavigate();
  const { user } = useAuth() as any;

  const [mode, setMode] = useState<Mode>("ESTIMATE");
  const [projectName, setProjectName] = useState("");

  const [estimateId, setEstimateId] = useState<number | "">("");
  const [estimates, setEstimates] = useState<EstimateRow[]>([]);
  const [loadingEst, setLoadingEst] = useState(false);
  const [errorEst, setErrorEst] = useState<string | null>(null);

  const [lines, setLines] = useState<MaterialLine[]>([]);
  const [loadingLines, setLoadingLines] = useState(false);
  const [errorLines, setErrorLines] = useState<string | null>(null);

  // 제품 선택 모달
  const [productModalOpen, setProductModalOpen] = useState(false);
  const [productKeyword, setProductKeyword] = useState("");
  const [productLoading, setProductLoading] = useState(false);
  const [productError, setProductError] = useState<string | null>(null);
  const [products, setProducts] = useState<ProductRow[]>([]);

  const estimateLines = lines.filter((x) => x.source === "ESTIMATE");
  const uplinkProductLines = lines.filter((x) => x.source === "UPLINK_PRODUCT");
  const manualLines = lines.filter((x) => x.source === "MANUAL");

  async function loadEstimatesOngoing() {
    setLoadingEst(true);
    setErrorEst(null);
    try {
      const qs = new URLSearchParams();
      qs.set("status", "ONGOING");
      const res = await api<any>(`/api/estimates?${qs.toString()}`);

      const items = Array.isArray(res) ? res : Array.isArray(res?.items) ? res.items : [];
      const normalized: EstimateRow[] = items
        .map((x: any) => ({
          id: Number(x?.id),
          title: x?.title ?? null,
          project_name: x?.project_name ?? x?.projectName ?? null,
          client_name: x?.client_name ?? x?.clientName ?? null,
          created_at: x?.created_at ?? x?.createdAt ?? null,
        }))
        .filter((x: any) => Number.isFinite(x.id));

      setEstimates(normalized);
    } catch (e: any) {
      setEstimates([]);
      setErrorEst(e?.message || "진행중 견적서 목록을 불러오지 못했습니다.");
    } finally {
      setLoadingEst(false);
    }
  }

  async function loadEstimateDetailAndFill(estimateIdNum: number) {
    setLoadingLines(true);
    setErrorLines(null);
    try {
      const detail = await api<any>(`/api/estimates/${estimateIdNum}`);

      const extracted = extractMaterialLinesFromEstimateDetail(detail);

      setLines((prev) => [
        ...extracted,
        ...prev.filter((x) => x.source === "UPLINK_PRODUCT"),
        ...prev.filter((x) => x.source === "MANUAL"),
      ]);

      const fallbackName =
        (detail?.title ?? detail?.project_name ?? detail?.projectName ?? "")?.toString().trim() || `견적서 #${estimateIdNum}`;
      if (!projectName.trim()) setProjectName(fallbackName);

      if (extracted.length === 0) setErrorLines("견적서 재료비(MATERIAL) 섹션에서 자재 라인을 찾지 못했습니다.");
    } catch (e: any) {
      setErrorLines(e?.message || "견적서 상세를 불러오지 못했습니다.");
    } finally {
      setLoadingLines(false);
    }
  }

  function onSelectEstimate(nextId: number | "") {
    setEstimateId(nextId);
    if (nextId === "") return;

    const found = estimates.find((x) => x.id === nextId);
    if (found) {
      const name =
        (found.project_name && found.project_name.trim()) ||
        (found.title && found.title.trim()) ||
        `견적서 #${found.id}`;
      setProjectName(name);
    }

    loadEstimateDetailAndFill(nextId);
  }

  function addManualLine() {
    setLines((prev) => [
      ...prev,
      {
        id: mkId(),
        source: "MANUAL",
        product_id: null,
        name: "",
        spec: "",
        unit: "",
        qty: 0,
        stock_qty: null,
        used_qty: null,
        prep_status: "PREPARING",
        note: "",
      },
    ]);
  }

  function updateLine(id: string, patch: Partial<MaterialLine>) {
    setLines((prev) => prev.map((x) => (x.id === id ? { ...x, ...patch } : x)));
  }

  function removeLine(id: string) {
    setLines((prev) => prev.filter((x) => x.id !== id));
  }

  async function loadProducts(keywordOverride?: string) {
    const kw = (keywordOverride ?? productKeyword).trim();

    setProductLoading(true);
    setProductError(null);

    try {
      const tryUrls = [
        `/api/products?keyword=${encodeURIComponent(kw)}`,
        `/api/products?q=${encodeURIComponent(kw)}`,
        `/api/products`,
      ];

      let res: any = null;
      let lastErr: any = null;

      for (const url of tryUrls) {
        try {
          res = await api<any>(url);
          lastErr = null;
          break;
        } catch (e) {
          lastErr = e;
        }
      }

      if (lastErr) throw lastErr;

      const items = Array.isArray(res) ? res : Array.isArray(res?.items) ? res.items : Array.isArray(res?.data) ? res.data : [];
      const normalized: ProductRow[] = items
        .map((x: any) => ({
          id: Number(x?.id),
          name: (x?.name ?? x?.product_name ?? x?.productName ?? "").toString(),
          spec: x?.spec ?? x?.specification ?? x?.product_spec ?? null,
          unit: x?.unit ?? x?.unit_name ?? x?.unitName ?? null,
        }))
        .filter((x: any) => Number.isFinite(x.id) && x.name);

      const lower = kw.toLowerCase();
      const filtered = lower ? normalized.filter((p) => `${p.name} ${p.spec ?? ""}`.toLowerCase().includes(lower)) : normalized;

      setProducts(filtered);
    } catch (e: any) {
      setProducts([]);
      setProductError(e?.message || "제품 리스트를 불러오지 못했습니다.");
    } finally {
      setProductLoading(false);
    }
  }

  function openProductModal() {
    setProductModalOpen(true);
    loadProducts("");
  }

  // 자동 검색(디바운스)
  useEffect(() => {
    if (!productModalOpen) return;

    const t = window.setTimeout(() => {
      loadProducts(productKeyword);
    }, 300);

    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [productKeyword, productModalOpen]);

  function addUplinkProductAndClose(p: ProductRow) {
    setLines((prev) => [
      ...prev,
      {
        id: mkId(),
        source: "UPLINK_PRODUCT",
        product_id: p.id,
        name: p.name,
        spec: (p.spec ?? "").toString(),
        unit: (p.unit ?? "EA").toString(),
        qty: 1,
        stock_qty: null,
        used_qty: null,
        prep_status: "PREPARING",
        note: "",
      },
    ]);

    setProductModalOpen(false);
  }

  useEffect(() => {
    loadEstimatesOngoing();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (mode === "MANUAL") setEstimateId("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  const hasAnyLines = estimateLines.length + uplinkProductLines.length + manualLines.length > 0;

  function GroupRow({ title }: { title: string }) {
    return (
      <tr>
        <td colSpan={8} style={{ padding: 10, fontWeight: 900, background: "rgba(255,255,255,0.06)" }}>
          {title}
        </td>
      </tr>
    );
  }

  const productListTitle = useMemo(() => {
    const kw = productKeyword.trim();
    if (productLoading) return "검색중…";
    if (kw) return `검색 결과 · ${products.length}건`;
    return `전체 · ${products.length}건`;
  }, [productKeyword, productLoading, products.length]);

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
        <div>
          <div style={{ fontSize: 20, fontWeight: 800 }}>자재 요청 신규 등록</div>
          <div style={{ marginTop: 6, fontSize: 13, opacity: 0.75 }}>
            견적서 재료비(자동입력) + 업링크 제품 + 수동 추가
          </div>
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <button
            type="button"
            onClick={() => navigate("/materials")}
            style={{
              padding: "10px 12px",
              borderRadius: 10,
              border: "1px solid rgba(255,255,255,0.15)",
              background: "rgba(255,255,255,0.06)",
              color: "white",
              cursor: "pointer",
              fontWeight: 800,
            }}
          >
            목록으로
          </button>
        </div>
      </div>

      {/* 자재요청사업명 + 등록자 */}
      <div style={{ marginTop: 14, padding: 12, borderRadius: 14, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.04)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <div style={{ fontSize: 12, opacity: 0.85, fontWeight: 800 }}>자재요청사업명</div>
            <div style={{ fontSize: 12, opacity: 0.8 }}>
              등록자: <span style={{ fontWeight: 900 }}>{user?.name ?? "-"}</span>
            </div>
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            <button
              type="button"
              onClick={() => setMode("ESTIMATE")}
              style={{
                padding: "8px 10px",
                borderRadius: 999,
                border: "1px solid rgba(255,255,255,0.15)",
                background: mode === "ESTIMATE" ? "rgba(255,255,255,0.16)" : "rgba(255,255,255,0.06)",
                color: "white",
                cursor: "pointer",
                fontWeight: 800,
                fontSize: 12,
              }}
            >
              견적서 선택
            </button>

            <button
              type="button"
              onClick={() => setMode("MANUAL")}
              style={{
                padding: "8px 10px",
                borderRadius: 999,
                border: "1px solid rgba(255,255,255,0.15)",
                background: mode === "MANUAL" ? "rgba(255,255,255,0.16)" : "rgba(255,255,255,0.06)",
                color: "white",
                cursor: "pointer",
                fontWeight: 800,
                fontSize: 12,
              }}
            >
              수동 입력
            </button>
          </div>
        </div>

        {mode === "ESTIMATE" && (
          <div style={{ marginTop: 10 }}>
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <select
                value={estimateId}
                onChange={(e) => onSelectEstimate(e.target.value ? Number(e.target.value) : "")}
                disabled={loadingEst}
                style={{
                  minWidth: 320,
                  padding: "10px 12px",
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.15)",
                  background: "rgba(17,24,39,0.7)",
                  color: "white",
                  outline: "none",
                  opacity: loadingEst ? 0.7 : 1,
                }}
              >
                <option value="">진행중 견적서를 선택하세요</option>
                {estimates.map((e) => (
                  <option key={e.id} value={e.id}>
                    {labelOfEstimate(e)}
                  </option>
                ))}
              </select>

              <button
                type="button"
                onClick={loadEstimatesOngoing}
                style={{
                  padding: "10px 12px",
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.15)",
                  background: "rgba(255,255,255,0.06)",
                  color: "white",
                  cursor: "pointer",
                  fontWeight: 800,
                }}
              >
                {loadingEst ? "불러오는 중…" : "새로고침"}
              </button>
            </div>

            {errorEst && <div style={{ marginTop: 10, fontSize: 12, color: "#ffb4b4" }}>{errorEst}</div>}
          </div>
        )}

        <input
          value={projectName}
          onChange={(e) => setProjectName(e.target.value)}
          placeholder={mode === "MANUAL" ? "자재요청사업명을 입력하세요" : "견적서 선택 시 자동 입력됩니다 (편집 가능)"}
          style={{
            width: "100%",
            marginTop: 12,
            padding: "10px 12px",
            borderRadius: 10,
            border: "1px solid rgba(255,255,255,0.15)",
            background: "rgba(17,24,39,0.7)",
            color: "white",
            outline: "none",
          }}
        />
      </div>

      {/* 자재 리스트 */}
      <div style={{ marginTop: 14, padding: 12, borderRadius: 14, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.04)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
          <div style={{ fontWeight: 800 }}>자재 리스트</div>

          <div style={{ display: "flex", gap: 8 }}>
            <button
              type="button"
              onClick={openProductModal}
              style={{
                padding: "10px 12px",
                borderRadius: 10,
                border: "1px solid rgba(255,255,255,0.15)",
                background: "rgba(255,255,255,0.06)",
                color: "white",
                cursor: "pointer",
                fontWeight: 900,
              }}
            >
              업링크 제품 추가
            </button>

            <button
              type="button"
              onClick={addManualLine}
              style={{
                padding: "10px 12px",
                borderRadius: 10,
                border: "1px solid rgba(255,255,255,0.15)",
                background: "rgba(255,255,255,0.06)",
                color: "white",
                cursor: "pointer",
                fontWeight: 900,
              }}
            >
              수동 추가
            </button>
          </div>
        </div>

        {loadingLines && <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>불러오는 중…</div>}
        {errorLines && <div style={{ marginTop: 10, fontSize: 12, color: "#ffb4b4" }}>{errorLines}</div>}

        <div style={{ marginTop: 12, width: "100%", overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "rgba(255,255,255,0.03)" }}>
                <th style={{ textAlign: "left", padding: 10, fontSize: 12, opacity: 0.85 }}>자재명</th>
                <th style={{ textAlign: "left", padding: 10, fontSize: 12, opacity: 0.85 }}>규격</th>
                <th style={{ textAlign: "left", padding: 10, fontSize: 12, opacity: 0.85, width: 90 }}>단위</th>
                <th style={{ textAlign: "right", padding: 10, fontSize: 12, opacity: 0.85, width: 110 }}>요청수량</th>
                <th style={{ textAlign: "right", padding: 10, fontSize: 12, opacity: 0.85, width: 110 }}>재고수량</th>
                <th style={{ textAlign: "right", padding: 10, fontSize: 12, opacity: 0.85, width: 110 }}>사용수량</th>
                <th style={{ textAlign: "center", padding: 10, fontSize: 12, opacity: 0.85, width: 120 }}>준비상황</th>
                <th style={{ textAlign: "left", padding: 10, fontSize: 12, opacity: 0.85, width: 220 }}>비고</th>
              </tr>
            </thead>

            <tbody>
              {!hasAnyLines && !loadingLines && (
                <tr>
                  <td colSpan={8} style={{ padding: 12, opacity: 0.75 }}>
                    견적서를 선택하면 재료비(자동 입력)가 표시됩니다. 또는 “업링크 제품 추가 / 수동 추가”로 항목을 넣으세요.
                  </td>
                </tr>
              )}

              {estimateLines.length > 0 && <GroupRow title={`견적서 재료비(자동 입력) · ${estimateLines.length}건`} />}
              {estimateLines.map((ln) => (
                <tr key={ln.id} style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                  <td style={{ padding: 10 }}>
                    <input
                      value={ln.name}
                      onChange={(e) => updateLine(ln.id, { name: e.target.value })}
                      style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }}
                    />
                  </td>
                  <td style={{ padding: 10 }}>
                    <input
                      value={ln.spec}
                      onChange={(e) => updateLine(ln.id, { spec: e.target.value })}
                      style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }}
                    />
                  </td>
                  <td style={{ padding: 10 }}>
                    <input
                      value={ln.unit}
                      onChange={(e) => updateLine(ln.id, { unit: e.target.value })}
                      style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }}
                    />
                  </td>
                  <td style={{ padding: 10 }}>
                    <input
                      value={Number.isFinite(ln.qty) ? String(ln.qty) : "0"}
                      onChange={(e) => updateLine(ln.id, { qty: Number(e.target.value) || 0 })}
                      inputMode="decimal"
                      style={{ width: "100%", textAlign: "right", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }}
                    />
                  </td>
                  <td style={{ padding: 10, textAlign: "right", opacity: 0.85 }}>{ln.stock_qty ?? "-"}</td>
                  <td style={{ padding: 10, textAlign: "right", opacity: 0.85 }}>{ln.used_qty ?? "-"}</td>
                  <td style={{ padding: 10, textAlign: "center" }}>
                    <select
                      value={ln.prep_status}
                      onChange={(e) => updateLine(ln.id, { prep_status: e.target.value as PrepStatus })}
                      style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)", background: "rgba(17,24,39,0.7)", color: "white" }}
                    >
                      <option value="PREPARING">준비중</option>
                      <option value="READY">준비완료</option>
                    </select>
                  </td>
                  <td style={{ padding: 10 }}>
                    <input
                      value={ln.note}
                      onChange={(e) => updateLine(ln.id, { note: e.target.value })}
                      placeholder="비고"
                      style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }}
                    />
                  </td>
                </tr>
              ))}

              {uplinkProductLines.length > 0 && <GroupRow title={`업링크 제품 · ${uplinkProductLines.length}건`} />}
              {uplinkProductLines.map((ln) => (
                <tr key={ln.id} style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                  <td style={{ padding: 10 }}>
                    <input value={ln.name} onChange={(e) => updateLine(ln.id, { name: e.target.value })} style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }} />
                  </td>
                  <td style={{ padding: 10 }}>
                    <input value={ln.spec} onChange={(e) => updateLine(ln.id, { spec: e.target.value })} style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }} />
                  </td>
                  <td style={{ padding: 10 }}>
                    <input value={ln.unit} onChange={(e) => updateLine(ln.id, { unit: e.target.value })} style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }} />
                  </td>
                  <td style={{ padding: 10 }}>
                    <input value={Number.isFinite(ln.qty) ? String(ln.qty) : "0"} onChange={(e) => updateLine(ln.id, { qty: Number(e.target.value) || 0 })} inputMode="decimal" style={{ width: "100%", textAlign: "right", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }} />
                  </td>
                  <td style={{ padding: 10, textAlign: "right", opacity: 0.85 }}>{ln.stock_qty ?? "-"}</td>
                  <td style={{ padding: 10, textAlign: "right", opacity: 0.85 }}>{ln.used_qty ?? "-"}</td>
                  <td style={{ padding: 10, textAlign: "center" }}>
                    <select value={ln.prep_status} onChange={(e) => updateLine(ln.id, { prep_status: e.target.value as PrepStatus })} style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)", background: "rgba(17,24,39,0.7)", color: "white" }}>
                      <option value="PREPARING">준비중</option>
                      <option value="READY">준비완료</option>
                    </select>
                  </td>
                  <td style={{ padding: 10 }}>
                    <input value={ln.note} onChange={(e) => updateLine(ln.id, { note: e.target.value })} placeholder="비고" style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }} />
                  </td>
                </tr>
              ))}

              {manualLines.length > 0 && <GroupRow title={`수동 추가 · ${manualLines.length}건`} />}
              {manualLines.map((ln) => (
                <tr key={ln.id} style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                  <td style={{ padding: 10 }}>
                    <input value={ln.name} onChange={(e) => updateLine(ln.id, { name: e.target.value })} placeholder="자재명" style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }} />
                  </td>
                  <td style={{ padding: 10 }}>
                    <input value={ln.spec} onChange={(e) => updateLine(ln.id, { spec: e.target.value })} placeholder="규격" style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }} />
                  </td>
                  <td style={{ padding: 10 }}>
                    <input value={ln.unit} onChange={(e) => updateLine(ln.id, { unit: e.target.value })} placeholder="단위" style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }} />
                  </td>
                  <td style={{ padding: 10 }}>
                    <input value={Number.isFinite(ln.qty) ? String(ln.qty) : "0"} onChange={(e) => updateLine(ln.id, { qty: Number(e.target.value) || 0 })} placeholder="0" inputMode="decimal" style={{ width: "100%", textAlign: "right", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }} />
                  </td>
                  <td style={{ padding: 10, textAlign: "right", opacity: 0.85 }}>{ln.stock_qty ?? "-"}</td>
                  <td style={{ padding: 10, textAlign: "right", opacity: 0.85 }}>{ln.used_qty ?? "-"}</td>
                  <td style={{ padding: 10, textAlign: "center" }}>
                    <select value={ln.prep_status} onChange={(e) => updateLine(ln.id, { prep_status: e.target.value as PrepStatus })} style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)", background: "rgba(17,24,39,0.7)", color: "white" }}>
                      <option value="PREPARING">준비중</option>
                      <option value="READY">준비완료</option>
                    </select>
                  </td>
                  <td style={{ padding: 10 }}>
                    <input value={ln.note} onChange={(e) => updateLine(ln.id, { note: e.target.value })} placeholder="비고" style={{ width: "100%", padding: "8px 10px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(17,24,39,0.7)", color: "white", outline: "none" }} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: 10, fontSize: 12, opacity: 0.75 }}>
          · 구분 순서: 견적서 재료비(자동입력) → 업링크 제품 → 수동 추가 (표 맨 앞 구분 컬럼은 제거됨)
        </div>
      </div>

      {/* 제품 선택 모달 */}
      {productModalOpen && (
        <div
          onClick={() => setProductModalOpen(false)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.55)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 16,
            zIndex: 50,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: 860,
              height: 640,
              overflow: "hidden",
              borderRadius: 16,
              border: "1px solid rgba(255,255,255,0.14)",
              background: "rgba(17,24,39,0.98)",
              color: "white",
              padding: 14,
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
              <div style={{ fontWeight: 900, fontSize: 16 }}>업링크 제품 선택</div>
              <button
                type="button"
                onClick={() => setProductModalOpen(false)}
                style={{
                  padding: "8px 10px",
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.15)",
                  background: "rgba(255,255,255,0.06)",
                  color: "white",
                  cursor: "pointer",
                  fontWeight: 900,
                }}
              >
                닫기
              </button>
            </div>

            <div style={{ marginTop: 12 }}>
              <input
                value={productKeyword}
                onChange={(e) => setProductKeyword(e.target.value)}
                placeholder="제품 검색(이름/규격) - 입력하면 자동 검색됩니다"
                style={{
                  width: "100%",
                  padding: "10px 12px",
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.15)",
                  background: "rgba(0,0,0,0.25)",
                  color: "white",
                  outline: "none",
                }}
              />
              <div style={{ marginTop: 8, fontSize: 12, opacity: 0.75 }}>{productListTitle}</div>
            </div>

            {productError && <div style={{ marginTop: 10, fontSize: 12, color: "#ffb4b4", whiteSpace: "pre-wrap" }}>{productError}</div>}

            <div style={{ marginTop: 12, flex: 1, overflow: "auto", paddingRight: 4 }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {!productLoading && products.length === 0 && (
                  <div style={{ padding: 12, borderRadius: 12, border: "1px solid rgba(255,255,255,0.12)", opacity: 0.75 }}>
                    표시할 제품이 없습니다.
                  </div>
                )}

                {products.map((p) => (
                  <div
                    key={p.id}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 12,
                      padding: 12,
                      borderRadius: 14,
                      border: "1px solid rgba(255,255,255,0.12)",
                      background: "rgba(255,255,255,0.04)",
                    }}
                  >
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontWeight: 900, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {p.name}
                      </div>
                      <div style={{ marginTop: 4, fontSize: 12, opacity: 0.85, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {p.spec ?? "-"} · 단위: {p.unit ?? "EA"}
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => addUplinkProductAndClose(p)}
                      style={{
                        flexShrink: 0,
                        padding: "10px 12px",
                        borderRadius: 12,
                        border: "1px solid rgba(255,255,255,0.15)",
                        background: "rgba(255,255,255,0.10)",
                        color: "white",
                        cursor: "pointer",
                        fontWeight: 900,
                      }}
                    >
                      추가
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ marginTop: 10, fontSize: 12, opacity: 0.75 }}>
              · “추가”를 누르면 즉시 모달이 닫히며, “업링크 제품” 그룹에 1건만 추가됩니다.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
