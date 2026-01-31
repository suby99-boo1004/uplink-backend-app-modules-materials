import React from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./lib/auth";

// Layout
import AppShell from "./layout/AppShell";

// Pages
import LoginPage from "./pages/LoginPage";
import DashboardPage from "./pages/DashboardPage";
import PlaceholderPage from "./pages/PlaceholderPage";

import AttendancePage from "./pages/AttendancePage";
import AttendanceMonthlyPage from "./pages/AttendanceMonthlyPage";
import AttendanceReportPage from "./pages/AttendanceReportPage";
import AdminAttendanceReportPage from "./pages/AdminAttendanceReportPage";

import ProductsPage from "./pages/Products/ProductsPage";

// Estimates
import EstimatesPage from "./pages/Estimates/EstimatesPage";
import EstimateDetailPage from "./pages/Estimates/EstimateDetailPage";
import EstimateRegisterPage from "./pages/Estimates/EstimateRegisterPage";
import EstimateEditPage from "./pages/Estimates/EstimateEditPage";

// ✅ Materials
import MaterialRequestsPage from "./pages/Materials/MaterialRequestsPage";
import MaterialRequestNewPage from "./pages/Materials/MaterialRequestNewPage";
import MaterialRequestDetailPage from "./pages/Materials/MaterialRequestDetailPage";

// Projects
import ProjectsPage from "./pages/Projects/ProjectsPage";
import ProjectDetailPage from "./pages/Projects/ProjectDetailPage";
import ProjectSearchPage from "./pages/Projects/ProjectSearchPage";

// Admin
import AdminDepartmentsPage from "./pages/AdminDepartmentsPage";
import AdminProjectsMetaPage from "./pages/AdminProjectsMetaPage";
import UserManagementPage from "./pages/Admin/UserManagementPage";
import StaffManagementPage from "./pages/Admin/StaffManagementPage";

function EmptyPage() {
  // 요구사항: 메뉴 클릭 시 우측에 아무것도 표시하지 않음
  return null;
}

function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth() as any;
  if (loading) return null;
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function LoginRedirectOrLogin() {
  const { user, loading } = useAuth() as any;
  if (loading) return null;
  if (user) return <Navigate to="/" replace />;
  return <LoginPage />;
}

function AdminOnly({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth() as any;
  if (loading) return null;

  const roleId = user?.role_id ?? null;
  // 기존 정책 유지: role_id === 6 만 관리자
  if (roleId !== 6) return <Navigate to="/" replace />;

  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      {/* Login */}
      <Route path="/login" element={<LoginRedirectOrLogin />} />

      {/* Shell */}
      <Route
        path="/"
        element={
          <RequireAuth>
            <AppShell />
          </RequireAuth>
        }
      >
        {/* 기본 홈 */}
        <Route index element={<DashboardPage />} />

        {/* 기존 라우트들 */}
        <Route path="dashboard" element={<DashboardPage />} />

        <Route path="attendance" element={<AttendancePage />} />
        <Route path="attendance/monthly" element={<AttendanceMonthlyPage />} />
        <Route path="attendance/report" element={<AttendanceReportPage />} />
        <Route
          path="admin/attendance-report"
          element={
            <AdminOnly>
              <AdminAttendanceReportPage />
            </AdminOnly>
          }
        />

        <Route path="products" element={<ProductsPage />} />

        {/* Estimates */}
        <Route path="estimates" element={<EstimatesPage />} />
        <Route path="estimates/new" element={<EstimateRegisterPage />} />
        <Route path="estimates/:estimateId" element={<EstimateDetailPage />} />
        <Route path="estimates/:estimateId/edit" element={<EstimateEditPage />} />

        {/* ✅ Materials */}
        <Route path="materials" element={<MaterialRequestsPage />} />
        <Route path="materials/new" element={<MaterialRequestNewPage />} />
        <Route path="materials/:id" element={<MaterialRequestDetailPage />} />

        {/* Disabled pages: keep menu, render nothing */}
        <Route path="maintenance" element={<EmptyPage />} />
        <Route path="design" element={<EmptyPage />} />
        <Route path="ideas" element={<PlaceholderPage />} />

        {/* Projects */}
        <Route path="projects" element={<ProjectsPage />} />
        <Route path="projects/search" element={<ProjectSearchPage />} />
        <Route path="projects/:projectId" element={<ProjectDetailPage />} />

        {/* Admin */}
        <Route
          path="admin/departments"
          element={
            <AdminOnly>
              <AdminDepartmentsPage />
            </AdminOnly>
          }
        />
        <Route
          path="admin/projects"
          element={
            <AdminOnly>
              <AdminProjectsMetaPage />
            </AdminOnly>
          }
        />
        <Route
          path="admin/users"
          element={
            <AdminOnly>
              <UserManagementPage />
            </AdminOnly>
          }
        />
        <Route
          path="admin/staff-management"
          element={
            <AdminOnly>
              <StaffManagementPage />
            </AdminOnly>
          }
        />

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
