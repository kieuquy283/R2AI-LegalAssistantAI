import { HugeiconsIcon } from "@hugeicons/react";
import {
  ChatIcon,
  ChartAnalysisIcon,
  Settings01Icon,
  UserIcon,
  Delete02Icon,
} from "@hugeicons/core-free-icons";
import { ChatSession, TabKey } from "../types";
import meetmeIcon from "../imgs/meetme.png";
import iconMassage from "../imgs/icon_massage.png";

type SidebarProps = {
  activeTab: TabKey;
  setActiveTab: (tab: TabKey) => void;
  sessions: ChatSession[];
  activeSessionId: string;
  onLoadSession: (id: string) => void;
  onCreateNewSession: () => void;
  onDeleteSession: (id: string) => void;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  chatDropdownOpen: boolean;
  setChatDropdownOpen: (open: boolean) => void;
  mobileOpen: boolean;
  setMobileOpen: (open: boolean) => void;
  theme: "light" | "dark";
  setTheme: (theme: "light" | "dark") => void;
};

export default function Sidebar({
  activeTab,
  setActiveTab,
  sessions,
  activeSessionId,
  onLoadSession,
  onCreateNewSession,
  onDeleteSession,
  sidebarOpen,
  setSidebarOpen,
  chatDropdownOpen,
  setChatDropdownOpen,
  mobileOpen,
  setMobileOpen,
  theme,
  setTheme,
}: SidebarProps) {
  const handleNavClick = (tab: TabKey) => {
    setActiveTab(tab);
    if (tab === "chat") {
      setChatDropdownOpen(!chatDropdownOpen);
    } else {
      setChatDropdownOpen(false);
    }
    setMobileOpen(false);
  };

  const handleSessionClick = (sessionId: string) => {
    onLoadSession(sessionId);
    setMobileOpen(false);
  };

  const toggleTheme = () => {
    setTheme(theme === "light" ? "dark" : "light");
  };

  return (
    <>
      {/* Mobile Drawer Backdrop */}
      {mobileOpen && (
        <div className="sidebar-overlay" onClick={() => setMobileOpen(false)} />
      )}

      <aside className={`sidebar ${sidebarOpen ? "open" : "collapsed"} ${mobileOpen ? "mobile-open" : ""}`}>
        {/* Top Header & New Chat button */}
        <div className="sidebar-top">
          <div className="sidebar-brand">
            {/* Hamburger Button inside Sidebar */}
            <button
              className="sidebar-toggle"
              type="button"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Thu gọn/Mở rộng menu"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12h18M3 6h18M3 18h18" strokeLinecap="round" />
              </svg>
            </button>
            <div className="brand-info">
              <img src={meetmeIcon} alt="GROUP5 RAG Logo" style={{ width: "24px", height: "24px", borderRadius: "4px", flexShrink: 0 }} />
              <span className="brand-title">GROUP5 RAG</span>
            </div>
          </div>

          <button
            className="new-chat-button"
            onClick={() => {
              onCreateNewSession();
              setMobileOpen(false);
            }}
          >
            <span className="new-chat-icon">+</span>
            <span className="new-chat-label">Cuộc trò chuyện mới</span>
          </button>
        </div>

        {/* Navigation & Recent sessions */}
        <div className="sidebar-section">
          <nav className="sidebar-nav">
            <button
              type="button"
              className={activeTab === "chat" ? "nav-button active" : "nav-button"}
              onClick={() => handleNavClick("chat")}
            >
              <span className="nav-icon">
                <HugeiconsIcon icon={ChatIcon} size={18} />
              </span>
              <span className="nav-label">Trò chuyện</span>
            </button>

            {chatDropdownOpen && (
              <div className="session-dropdown">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className={session.id === activeSessionId ? "session-item-wrapper active" : "session-item-wrapper"}
                  >
                    <button
                      type="button"
                      className="session-item"
                      onClick={() => handleSessionClick(session.id)}
                      title={session.name}
                    >
                      <img src={iconMassage} alt="Session" className="chat-bubble-mini-icon" style={{ width: "16px", height: "16px", marginRight: "6px", verticalAlign: "middle" }} />
                      <span className="session-text">{session.name}</span>
                    </button>
                    <button
                      type="button"
                      className="session-delete-button"
                      onClick={(event) => {
                        event.stopPropagation();
                        onDeleteSession(session.id);
                      }}
                      aria-label={`Xóa ${session.name}`}
                    >
                      <HugeiconsIcon icon={Delete02Icon} size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <button
              type="button"
              className={activeTab === "evaluation" ? "nav-button active" : "nav-button"}
              onClick={() => handleNavClick("evaluation")}
            >
              <span className="nav-icon">
                <HugeiconsIcon icon={ChartAnalysisIcon} size={18} />
              </span>
              <span className="nav-label">Tóm tắt đánh giá</span>
            </button>

            <button
              type="button"
              className={activeTab === "settings" ? "nav-button active" : "nav-button"}
              onClick={() => handleNavClick("settings")}
            >
              <span className="nav-icon">
                <HugeiconsIcon icon={Settings01Icon} size={18} />
              </span>
              <span className="nav-label">Cài đặt</span>
            </button>
          </nav>
        </div>

        {/* Footer with User and Theme Toggle */}
        <div className="sidebar-footer">
          <button className="theme-toggle-button nav-button" onClick={toggleTheme} type="button">
            <span className="nav-icon">
              {theme === "light" ? (
                // Moon Icon for switching to dark
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
              ) : (
                // Sun Icon for switching to light
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="5" />
                  <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
                </svg>
              )}
            </span>
            <span className="nav-label">{theme === "light" ? "Giao diện tối" : "Giao diện sáng"}</span>
          </button>

          <div className="profile-card">
            <div className="profile-icon">
              <HugeiconsIcon icon={UserIcon} size={16} />
            </div>
            <div className="profile-info">
              <div className="profile-name">Người dùng</div>
              <div className="profile-role">Miễn phí</div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
