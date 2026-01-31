import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../../lib/api";

type TabKey = "ONGOING" | "DONE" | "CANCELED";

type MRRow = {
  id: number;
  project_id: number | null;
  memo?: string | null;
  requested_by_name?: string | null;
  created_at?: string | null;
  prep_status?: "PREPARING" | "READY" | string | null;
  project_name?: string | null;
};

function fmtDate(s?: string | null) {
  if (!s) return "-";
  try {
    const d = new Date(s);
    if (Number.isNaN(d.getTime())) return s;
    const yy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    return `${yy}-${mm}-${dd}`;
  } catch {
    return s;
  }
}

function prepLabel(v?: string | null) {
  if (!v) return "-";
  if (v === "PREPARING") return "준비중";
  if (v === "READY") return "준비완료";
  return v;
}

export default function MaterialRequestsPage() {
  const navigate = useNavigate();

  const [tab, setTab] = useState<TabKey>("ONGOING");
  const [year, setYear] = useState<number>(new Date().getFullYear());
  const [keyword, setKeyword] = useState<string>("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<MRRow[]>([]);

  const tabs = useMemo(
    () => [
      { key: "ONGOING" as const, label: "진행중" },
      { key: "DONE" as const, label: "사업완료" },
      { key: "CANCELED" as const, label: "사업취소" },
    ],
    []
  );

  async function fetchList(next?: { tab?: TabKey; year?: number; keyword?: string }) {
    const t = next?.tab ?? tab;
    const y = next?.year ?? year;
    const q = next?.keyword ?? keyword;

    setLoading(true);
    setError(null);

    try {
      const qs = new URLSearchParams();
      if (y) qs.set("year", String(y));
      qs.set("state", t);
      if (q && q.trim()) qs.set("q", q.trim());

      const res = await api<{ items: MRRow[] }>(`/api/material-requests?${qs.toString()}`);
      setRows(Array.isArray(res.items) ? res.items : []);
    } catch (e: any) {
      setRows([]);
      setError(e?.message || "목록을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  function onClickNew() {
    // ✅ 이번 단계 목표: 신규 등록 라우팅
    navigate("/materials/new");
  }

  function onClickSearch() {
    fetchList();
  }

  useEffect(() => {
    fetchList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    fetchList({ tab });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
        <div>
          <div style={{ fontSize: 20, fontWeight: 800 }}>자재 요청</div>
          <div style={{ marginTop: 6, fontSize: 13, opacity: 0.75 }}>
            진행중/사업완료/사업취소 자재요청 리스트
          </div>
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <button
            type="button"
            onClick={onClickNew}
            style={{
              padding: "10px 12px",
              borderRadius: 10,
              border: "1px solid rgba(255,255,255,0.15)",
              background: "rgba(255,255,255,0.06)",
              color: "white",
              cursor: "pointer",
              fontWeight: 700,
            }}
          >
            신규 등록
          </button>
        </div>
      </div>

      {/* 탭 */}
      <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
        {tabs.map((t) => {
          const active = tab === t.key;
          return (
            <button
              key={t.key}
              type="button"
              onClick={() => setTab(t.key)}
              style={{
                padding: "10px 12px",
                borderRadius: 999,
                border: "1px solid rgba(255,255,255,0.15)",
                background: active ? "rgba(255,255,255,0.16)" : "rgba(255,255,255,0.06)",
                color: "white",
                cursor: "pointer",
                fontWeight: 800,
              }}
            >
              {t.label}
            </button>
          );
        })}
      </div>

      {/* 검색 영역 */}
      <div
        style={{
          marginTop: 14,
          padding: 12,
          borderRadius: 14,
          border: "1px solid rgba(255,255,255,0.12)",
          background: "rgba(255,255,255,0.04)",
        }}
      >
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <div style={{ fontSize: 12, opacity: 0.8, fontWeight: 700 }}>연도</div>
            <select
              value={year}
              onChange={(e) => setYear(Number(e.target.value))}
              style={{
                padding: "8px 10px",
                borderRadius: 10,
                border: "1px solid rgba(255,255,255,0.15)",
                background: "rgba(17,24,39,0.7)",
                color: "white",
                outline: "none",
              }}
            >
              {[year, year - 1, year - 2].map((y) => (
                <option key={y} value={y}>
                  {y}년
                </option>
              ))}
            </select>
          </div>

          <div style={{ flex: 1, minWidth: 220 }}>
            <input
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") onClickSearch();
              }}
              placeholder="검색(사업명/등록자)"
              style={{
                width: "100%",
                padding: "10px 12px",
                borderRadius: 10,
                border: "1px solid rgba(255,255,255,0.15)",
                background: "rgba(17,24,39,0.7)",
                color: "white",
                outline: "none",
              }}
            />
          </div>

          <button
            type="button"
            onClick={onClickSearch}
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
            검색
          </button>
        </div>
      </div>

      {/* 에러 */}
      {error && (
        <div
          style={{
            marginTop: 12,
            padding: 12,
            borderRadius: 12,
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,0,0,0.08)",
            whiteSpace: "pre-wrap",
          }}
        >
          <div style={{ fontWeight: 800 }}>불러오기 실패</div>
          <div style={{ marginTop: 6, fontSize: 12, opacity: 0.9 }}>{error}</div>
        </div>
      )}

      {/* 리스트 */}
      <div
        style={{
          marginTop: 14,
          borderRadius: 14,
          border: "1px solid rgba(255,255,255,0.12)",
          overflow: "hidden",
        }}
      >
        <div style={{ padding: 12, background: "rgba(255,255,255,0.06)", fontWeight: 800 }}>
          {tab === "ONGOING" && "진행중 자재 요청"}
          {tab === "DONE" && "사업완료 자재 요청"}
          {tab === "CANCELED" && "사업취소 자재 요청"}
          {loading && <span style={{ marginLeft: 10, fontSize: 12, opacity: 0.75 }}>불러오는 중…</span>}
        </div>

        <div style={{ width: "100%", overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "rgba(255,255,255,0.03)" }}>
                <th style={{ textAlign: "left", padding: 12, fontSize: 12, opacity: 0.85 }}>사업명</th>
                <th style={{ textAlign: "left", padding: 12, fontSize: 12, opacity: 0.85 }}>등록자</th>
                <th style={{ textAlign: "left", padding: 12, fontSize: 12, opacity: 0.85 }}>준비상태</th>
                <th style={{ textAlign: "left", padding: 12, fontSize: 12, opacity: 0.85 }}>등록일</th>
              </tr>
            </thead>
            <tbody>
              {!loading && rows.length === 0 && (
                <tr>
                  <td colSpan={4} style={{ padding: 16, opacity: 0.75 }}>
                    표시할 데이터가 없습니다.
                  </td>
                </tr>
              )}

              {rows.map((r) => {
                const title =
                  (r.project_name && r.project_name.trim()) ||
                  (r.memo && r.memo.trim()) ||
                  (r.project_id ? `프로젝트 #${r.project_id}` : `자재요청 #${r.id}`);

                return (
                  <tr key={r.id} style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                    <td style={{ padding: 12, fontWeight: 700 }}>{title}</td>
                    <td style={{ padding: 12, opacity: 0.9 }}>{r.requested_by_name || "-"}</td>
                    <td style={{ padding: 12 }}>
                      <span
                        style={{
                          padding: "6px 10px",
                          borderRadius: 999,
                          border: "1px solid rgba(255,255,255,0.15)",
                          background: r.prep_status === "READY" ? "rgba(255,255,255,0.14)" : "rgba(255,255,255,0.06)",
                          fontWeight: 800,
                          fontSize: 12,
                        }}
                      >
                        {prepLabel(r.prep_status || null)}
                      </span>
                    </td>
                    <td style={{ padding: 12, opacity: 0.85 }}>{fmtDate(r.created_at || null)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
