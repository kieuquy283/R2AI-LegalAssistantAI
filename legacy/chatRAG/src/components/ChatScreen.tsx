import { FormEvent, KeyboardEvent, MouseEvent, useEffect, useRef, useState } from "react";
import { Message } from "../types";
import meetmeIcon from "../imgs/meetme.png";

type ChatScreenProps = {
  messages: Message[];
  loading: boolean;
  question: string;
  setQuestion: (q: string) => void;
  onSendMessage: (
    event?: FormEvent<HTMLFormElement> | MouseEvent<HTMLButtonElement>
  ) => Promise<void>;
  activeSessionName: string;
  onOpenMobileMenu: () => void;
};

const SUGGESTIONS = [
  "Đất bị giải tỏa nhưng đang thế chấp thì xử lý thế nào?",
  "Tóm tắt các văn bản Nghị định 121/2026/NĐ-CP",
  "Cách thức hoạt động của tính năng Query Rewrite",
  "Xem báo cáo tóm tắt đánh giá hiệu suất RAG"
];

export default function ChatScreen({
  messages,
  loading,
  question,
  setQuestion,
  onSendMessage,
  activeSessionName,
  onOpenMobileMenu,
}: ChatScreenProps) {
  const [expandedMessages, setExpandedMessages] = useState<number[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Adjust textarea height dynamically
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [question]);

  const toggleMessageMetadata = (messageIndex: number) => {
    setExpandedMessages((prev) =>
      prev.includes(messageIndex)
        ? prev.filter((index) => index !== messageIndex)
        : [...prev, messageIndex]
    );
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void onSendMessage();
    }
  };

  const handleSuggestionClick = (suggestionText: string) => {
    setQuestion(suggestionText);
    textareaRef.current?.focus();
  };

  // Check if we are in the landing/welcome state
  // (i.e. we only have the default greeting, or no messages at all)
  const isWelcomeState =
    messages.length === 0 ||
    (messages.length === 1 &&
      messages[0].role === "assistant" &&
      (messages[0].content.includes("Xin chào") || messages[0].content.includes("chào")));

  return (
    <section className="chat-screen">
      {/* Top Header Bar */}
      <div className="chat-topbar">
        <button
          className="mobile-menu-trigger"
          type="button"
          onClick={onOpenMobileMenu}
          aria-label="Mở menu"
        >
          ☰
        </button>
        <div className="chat-title-container">
          <p className="chat-topic">{activeSessionName || "Trò chuyện"}</p>
          <span className="chat-model-badge">GROUP5 RAG v1.0</span>
        </div>
      </div>

      {/* Messages / Welcome Greeting Area */}
      <div className="chat-messages-container">
        {isWelcomeState ? (
          <div className="chat-welcome-screen">
            <div className="welcome-header">
              <h1 className="welcome-title-gradient">Xin chào, tôi là GROUP5 RAG</h1>
              <p className="welcome-subtitle">Tôi có thể giúp bạn tìm kiếm thông tin, tóm tắt đánh giá và tra cứu văn bản pháp luật.</p>
            </div>
            
            <div className="suggestions-grid">
              {SUGGESTIONS.map((sug, i) => (
                <button
                  key={i}
                  className="suggestion-card"
                  onClick={() => handleSuggestionClick(sug)}
                  type="button"
                >
                  <p className="suggestion-card-text">{sug}</p>
                  <span className="suggestion-arrow">↗</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="chat-messages">
            {messages.map((message, index) => (
              <div key={index} className={`message-row ${message.role}`}>
                <div className="message-avatar-column">
                  {message.role === "assistant" ? (
                    <div className="message-avatar assistant-avatar-spark">
                      <img src={meetmeIcon} alt="GROUP5 RAG" style={{ width: "28px", height: "28px", borderRadius: "50%" }} />
                    </div>
                  ) : (
                    <div className="message-avatar user-avatar-circle">Y</div>
                  )}
                </div>
                <div className="message-content-column">
                  <div className="message-header-gemini">
                    <span className="sender-name-gemini">
                      {message.role === "assistant" ? "GROUP5 RAG" : "Bạn"}
                    </span>
                    <span className="message-time-gemini">{message.time}</span>
                  </div>
                  <div className="message-body-gemini">{message.content}</div>

                  {message.role === "assistant" && message.metadata && (
                    <div className="message-meta-section">
                      {/* Source Citation Chips */}
                      {message.metadata.top_files && message.metadata.top_files.length > 0 && (
                        <div className="citations-inline">
                          <span className="citation-title">Nguồn đã dùng:</span>
                          <div className="citation-chips-row">
                            {message.metadata.top_files.map((file, fileIndex) => (
                              <button
                                key={fileIndex}
                                className="citation-chip"
                                onClick={() => toggleMessageMetadata(index)}
                                title={file.source_file}
                                type="button"
                              >
                                <span className="citation-chip-number">[{fileIndex + 1}]</span>
                                <span className="citation-chip-name">
                                  {file.source_file.split(/[/\\]/).pop()}
                                </span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="message-meta-actions-gemini">
                        <button
                          type="button"
                          className={`message-meta-toggle-gemini ${
                            expandedMessages.includes(index) ? "active" : ""
                          }`}
                          onClick={() => toggleMessageMetadata(index)}
                        >
                          {expandedMessages.includes(index) ? "Ẩn tham số RAG" : "Thông số RAG"}
                        </button>
                      </div>

                      {expandedMessages.includes(index) && (
                        <div className="message-meta-gemini">
                          {message.metadata.show_rewritten_query &&
                            message.metadata.rewritten_query && (
                              <div className="meta-item">
                                <span className="meta-item-key">Query đã viết lại:</span>
                                <span className="meta-item-val">{message.metadata.rewritten_query}</span>
                              </div>
                            )}
                          <div className="meta-item">
                            <span className="meta-item-key">Sử dụng Rewrite:</span>
                            <span className="meta-item-val">
                              {message.metadata.used_rewrite ? "Đã dùng" : "Không dùng"}
                            </span>
                          </div>
                          {message.metadata.mode && (
                            <div className="meta-item">
                              <span className="meta-item-key">Chế độ RAG:</span>
                              <span className="meta-item-val">{message.metadata.mode}</span>
                            </div>
                          )}
                          
                          {/* Expanded Files Citation Details */}
                          {message.metadata.top_files && message.metadata.top_files.length > 0 && (
                            <div className="meta-files-expanded">
                              <p className="meta-files-title-expanded">Chi tiết tài liệu:</p>
                              <div className="meta-files-list">
                                {message.metadata.top_files.map((file, fileIndex) => (
                                  <div key={fileIndex} className="meta-file-row">
                                    <span className="meta-file-idx">#{fileIndex + 1}</span>
                                    <span className="meta-file-name" title={file.source_file}>
                                      {file.source_file}
                                    </span>
                                    <span className="meta-file-score">
                                      Best Score: <strong>{file.best_score.toFixed(4)}</strong>
                                    </span>
                                    <span className="meta-file-hits">
                                      Hits: <strong>{file.hits}</strong>
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="message-row assistant">
                <div className="message-avatar-column">
                  <div className="message-avatar assistant-avatar-spark anim-pulse">
                    <img src={meetmeIcon} alt="GROUP5 RAG" style={{ width: "28px", height: "28px", borderRadius: "50%" }} />
                  </div>
                </div>
                <div className="message-content-column">
                  <div className="message-body-gemini">
                    <div className="gemini-loading-placeholder">
                      <div className="shimmer-bar short"></div>
                      <div className="shimmer-bar medium"></div>
                      <div className="shimmer-bar long"></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input panel & citations info */}
      <div className="chat-footer">
        <form
          className="input-panel-gemini"
          onSubmit={(e) => {
            e.preventDefault();
            void onSendMessage();
          }}
        >
          <div className="input-wrapper-gemini">
            <textarea
              ref={textareaRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Nhập câu hỏi hoặc yêu cầu tại đây..."
              rows={1}
            />
          </div>
          <button
            className="action-send-gemini"
            type="submit"
            disabled={loading || !question.trim()}
            aria-label="Gửi"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </form>
        <div className="chat-disclaimer">
          GROUP5 RAG có thể đưa ra thông tin không chính xác về pháp luật. Vui lòng xác minh lại thông tin quan trọng.
        </div>
      </div>
    </section>
  );
}
