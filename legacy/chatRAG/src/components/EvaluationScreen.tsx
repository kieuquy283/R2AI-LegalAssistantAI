import { EvaluationStats } from "../types";

type EvaluationScreenProps = {
  results: EvaluationStats[];
  loading: boolean;
  error: string;
  sessionTrendData: {
    sessionName: string;
    best_score: number;
    avg_score: number;
    hits: number;
  }[];
  trendMetric: "best_score" | "avg_score" | "hits";
  setTrendMetric: (metric: "best_score" | "avg_score" | "hits") => void;
  maxSessionTrendValue: number;
  onRefresh: () => void;
  onOpenMobileMenu: () => void;
};

export default function EvaluationScreen({
  results,
  loading,
  error,
  sessionTrendData,
  trendMetric,
  setTrendMetric,
  maxSessionTrendValue,
  onRefresh,
  onOpenMobileMenu,
}: EvaluationScreenProps) {
  return (
    <section className="evaluation-screen">
      {/* Mobile Header Bar */}
      <div className="chat-topbar evaluation-topbar">
        <button
          className="mobile-menu-trigger"
          type="button"
          onClick={onOpenMobileMenu}
          aria-label="Mở menu"
        >
          ☰
        </button>
        <div className="chat-title-container">
          <p className="chat-topic">Tóm tắt đánh giá</p>
        </div>
      </div>

      <div className="evaluation-header">
        <div className="evaluation-intro">
          <h1>Số liệu đánh giá hệ thống</h1>
          <p>So sánh độ chính xác và xu hướng tìm kiếm tài liệu của các phiên bản mô hình.</p>
        </div>
        <div className="evaluation-actions">
          <button
            className={`refresh-eval-button ${loading ? "loading" : ""}`}
            onClick={onRefresh}
            disabled={loading}
          >
            {loading ? "Đang tính toán..." : "Làm mới số liệu"}
          </button>
        </div>
      </div>

      {loading ? (
        <div className="empty-state">
          <div className="spinner"></div>
          <p>Đang tải dữ liệu đánh giá và mô hình...</p>
        </div>
      ) : error ? (
        <div className="warning-block">
          <h3>Lỗi tải dữ liệu</h3>
          <p>{error}</p>
        </div>
      ) : results.length === 0 ? (
        <div className="empty-state">
          <p>Không tìm thấy dữ liệu đánh giá. Vui lòng bấm làm mới số liệu.</p>
        </div>
      ) : (
        <>
          {/* Card list evaluation metrics */}
          <div className="evaluation-cards-row">
            {results.map((item) => (
              <div key={item.name} className="evaluation-card-gemini">
                <div className="card-header-badge">
                  <span className="card-label-gemini">{item.name}</span>
                  <span className="top-k-badge">Top-{item.top_k}</span>
                </div>

                <div className="card-metrics">
                  <div className="card-metric-row">
                    <span className="metric-label">Hit Rate</span>
                    <strong className="metric-value">{(item.hit * 100).toFixed(2)}%</strong>
                  </div>

                  <div className="card-metric-row">
                    <span className="metric-label">Recall</span>
                    <strong className="metric-value">{item.recall.toFixed(4)}</strong>
                  </div>

                  <div className="progress-bar-container">
                    <div
                      className="progress-fill-bar"
                      style={{ width: `${Math.min(item.recall * 100, 100)}%` }}
                    />
                  </div>

                  <div className="card-metric-row border-top">
                    <span className="metric-label">MRR Score</span>
                    <strong className="metric-value highlight-gemini">{item.mrr.toFixed(4)}</strong>
                  </div>
                </div>

                <div className="eval-path-footer">
                  <span>Mẫu thử: {item.sample_count}</span>
                  <span className="eval-path-text" title={item.eval_path}>
                    {item.eval_path.split(/[/\\]/).pop()}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Performance Trend chart */}
          <div className="performance-section-gemini">
            <div className="performance-header">
              <div className="performance-intro">
                <h2>Xu hướng cải thiện tìm kiếm</h2>
                <p>Thống kê chất lượng truy xuất tài liệu trung bình trên mỗi cuộc hội thoại.</p>
              </div>
              <div className="trend-filters">
                <button
                  type="button"
                  className={trendMetric === "avg_score" ? "filter-pill active" : "filter-pill"}
                  onClick={() => setTrendMetric("avg_score")}
                >
                  Avg Score
                </button>
                <button
                  type="button"
                  className={trendMetric === "best_score" ? "filter-pill active" : "filter-pill"}
                  onClick={() => setTrendMetric("best_score")}
                >
                  Best Score
                </button>
                <button
                  type="button"
                  className={trendMetric === "hits" ? "filter-pill active" : "filter-pill"}
                  onClick={() => setTrendMetric("hits")}
                >
                  Hits
                </button>
              </div>
            </div>

            <div className="trend-card-gemini">
              <div className="trend-chart-description">
                <span className="trend-stat-title">
                  Chỉ số: {trendMetric.replace("_", " ").toUpperCase()}
                </span>
              </div>

              <div className="trend-graph">
                {sessionTrendData.length > 0 ? (
                  sessionTrendData.map((item) => {
                    const value = item[trendMetric];
                    const barHeight = `${Math.max(
                      8,
                      Math.min((value / maxSessionTrendValue) * 100, 100)
                    )}%`;
                    return (
                      <div key={item.sessionName} className="trend-bar-wrapper">
                        <div className="trend-bar-gemini" style={{ height: barHeight }}>
                          <span className="trend-bar-value">
                            {trendMetric === "hits" ? value.toFixed(1) : value.toFixed(3)}
                          </span>
                        </div>
                        <div className="trend-bar-label" title={item.sessionName}>
                          {item.sessionName}
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="empty-state inline">
                    <p>Chưa lưu lịch sử hội thoại có dữ liệu truy xuất tài liệu.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
