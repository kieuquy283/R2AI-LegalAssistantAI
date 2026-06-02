type SettingsScreenProps = {
  onOpenMobileMenu: () => void;
};

export default function SettingsScreen({ onOpenMobileMenu }: SettingsScreenProps) {
  return (
    <section className="settings-screen">
      {/* Mobile Top Header */}
      <div className="chat-topbar settings-topbar">
        <button
          className="mobile-menu-trigger"
          type="button"
          onClick={onOpenMobileMenu}
          aria-label="Mở menu"
        >
          ☰
        </button>
        <div className="chat-title-container">
          <p className="chat-topic">Cài đặt</p>
        </div>
      </div>

      <div className="settings-header">
        <h1>Cấu hình ứng dụng</h1>
        <p>Căn chỉnh tham số mô hình RAG và liên kết dịch vụ backend của bạn.</p>
      </div>

      <div className="settings-grid">
        <div className="settings-card-gemini">
          <h3>Kết nối máy chủ</h3>
          <div className="settings-form">
            <div className="form-group">
              <label>FastAPI Base URL</label>
              <input
                type="text"
                defaultValue="http://127.0.0.1:8000"
                className="settings-input-gemini"
                disabled
              />
              <span className="form-tip">Địa chỉ IP kết nối cục bộ.</span>
            </div>

            <div className="form-group">
              <label>Mô hình ngôn ngữ (LLM)</label>
              <select className="settings-select-gemini" defaultValue="gemini" disabled>
                <option value="gemini">Google Gemini Pro (API)</option>
                <option value="openai">OpenAI GPT-4o (API)</option>
              </select>
            </div>
          </div>
        </div>

        <div className="settings-card-gemini">
          <h3>Cấu hình Truy xuất tài liệu</h3>
          <div className="settings-form">
            <div className="form-group inline-row">
              <label>Kích thước Top-K tài liệu</label>
              <input
                type="number"
                defaultValue={5}
                className="settings-input-gemini width-small"
                disabled
              />
            </div>
            <div className="form-group inline-row">
              <label>Tự động tối ưu hóa câu hỏi (Query Rewrite)</label>
              <input type="checkbox" defaultChecked className="settings-checkbox-gemini" disabled />
            </div>
            <div className="form-group">
              <label>Metric xếp hạng chính</label>
              <select className="settings-select-gemini" defaultValue="mrr" disabled>
                <option value="recall">Recall Score</option>
                <option value="hit">Hit Rate</option>
                <option value="mrr">MRR (Mean Reciprocal Rank)</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="info-block-gemini settings-info">
        <strong>Lưu ý cấu hình:</strong>
        <span>
          Các thiết lập này hiện tại được tải tự động từ file `.env` ở máy chủ. Việc cập nhật trực tiếp trên UI sẽ được kích hoạt ở các bản nâng cấp tiếp theo.
        </span>
      </div>
    </section>
  );
}
