import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../../lib/api";

type MRHeader = {
  id: number;
  memo?: string | null;
  status?: string | null;
  warehouse_id?: number | null;
  requested_by_name?: string | null;
  created_at?: string | null;
};

type MRItem = {
  id: number;
  item_name_snapshot: string;
  spec_snapshot: string;
  unit_snapshot: string;
  qty_requested: number;
  note?: string | null;
};

function fmtDateTime(s?: string | null) {
  if (!s) return "-";
  try {
    const d = new Date(s);
    if (Number.isNaN(d.getTime())) return s;
    const yy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    const hh = String(d.getHours()).padStart(2, "0");
    const mi = String(d.getMinutes()).padStart(2, "0");
    return `${yy}-${mm}-${dd} ${hh}:${mi}`;
  } catch {
    return s;
  }
}

export default function MaterialRequestDetailPage() {
  const navigate = useNavigate();
  const params = useParams();
  const mrId = Number(params.id);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [header, setHeader] = useState<MRHeader | null>(null);
  const [items, setItems] = useState<MRItem[]>([]);

  async function fetchDetail() {
    if (!mrId || Number.isNaN(mrId)) {
      setError("잘못된 자재요청 ID 입니다.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await api<{ header: MRHeader; items: MRItem[] }>(`/api/material-requests/${mrId}`);
      setHeader(res?.header ?? null);
      setItems(Array.isArray(res?.items) ? res.items : []);
    } catch (e: any) {
      setError(e?.message || "상세를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchDetail();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mrId]);

  const title = (header?.memo && header.memo.trim()) || `자재요청 #${mrId}`;

  return (
    <div style={{ padding: 18, color: "white" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
          <div>
            <div style={{ fontSize: 22, fontWeight: 900 }}>{title}</div>
            <div style={{ marginTop: 6, opacity: 0.85, fontSize: 13 }}>
              등록자: {header?.requested_by_name || "-"} · 등록일: {fmtDateTime(header?.created_at || null)} · 상태:{" "}
              {header?.status || "-"}
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
                fontWeight: 900,
              }}
            >
              목록으로
            </button>
          </div>
        </div>

        {error && (
          <div style={{ padding: 12, borderRadius: 12, border: "1px solid rgba(239,68,68,0.35)", background: "rgba(239,68,68,0.12)" }}>
            <div style={{ fontWeight: 900, marginBottom: 6 }}>불러오기 실패</div>
            <div style={{ opacity: 0.95, whiteSpace: "pre-wrap" }}>{error}</div>
          </div>
        )}

        <div style={{ marginTop: 12, borderRadius: 14, border: "1px solid rgba(255,255,255,0.10)", background: "rgba(255,255,255,0.03)", overflow: "hidden" }}>
          <div style={{ padding: "12px 14px", borderBottom: "1px solid rgba(255,255,255,0.08)", fontWeight: 900 }}>
            요청 항목 {loading ? "(불러오는 중…)" : `(${items.length})`}
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "rgba(255,255,255,0.03)" }}>
                  <th style={{ textAlign: "left", padding: 10, fontSize: 12, opacity: 0.85 }}>자재명</th>
                  <th style={{ textAlign: "left", padding: 10, fontSize: 12, opacity: 0.85 }}>규격</th>
                  <th style={{ textAlign: "left", padding: 10, fontSize: 12, opacity: 0.85, width: 90 }}>단위</th>
                  <th style={{ textAlign: "right", padding: 10, fontSize: 12, opacity: 0.85, width: 110 }}>요청수량</th>
                  <th style={{ textAlign: "left", padding: 10, fontSize: 12, opacity: 0.85, width: 220 }}>비고</th>
                </tr>
              </thead>
              <tbody>
                {!loading && items.length === 0 && (
                  <tr>
                    <td colSpan={5} style={{ padding: 16, opacity: 0.75 }}>
                      표시할 데이터가 없습니다.
                    </td>
                  </tr>
                )}
                {items.map((it) => (
                  <tr key={it.id} style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                    <td style={{ padding: 10, fontWeight: 700 }}>{it.item_name_snapshot || "-"}</td>
                    <td style={{ padding: 10, opacity: 0.9 }}>{it.spec_snapshot || "-"}</td>
                    <td style={{ padding: 10, opacity: 0.9 }}>{it.unit_snapshot || "-"}</td>
                    <td style={{ padding: 10, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{Number(it.qty_requested || 0)}</td>
                    <td style={{ padding: 10, opacity: 0.9 }}>{it.note || ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
