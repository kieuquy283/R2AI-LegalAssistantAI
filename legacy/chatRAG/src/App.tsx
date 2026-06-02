import { FormEvent, MouseEvent, useEffect, useState } from "react";
import { ChatSession, Message, TabKey, TopFileInfo, EvaluationResponse, EvaluationStats } from "./types";
import Sidebar from "./components/Sidebar";
import ChatScreen from "./components/ChatScreen";
import EvaluationScreen from "./components/EvaluationScreen";
import SettingsScreen from "./components/SettingsScreen";

const DEFAULT_MESSAGES: Message[] = [
  {
    role: "assistant",
    content: "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?",
  },
];

const SESSIONS_API_URL = "http://127.0.0.1:8000/sessions";

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("chat");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>(DEFAULT_MESSAGES);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [chatDropdownOpen, setChatDropdownOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState<EvaluationStats[]>([]);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");
  const [trendMetric, setTrendMetric] = useState<"best_score" | "avg_score" | "hits">("avg_score");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileOpen, setMobileOpen] = useState(false);
  
  // Theme state: default to dark or read from localStorage
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const saved = localStorage.getItem("theme");
    return saved === "light" ? "light" : "dark";
  });

  // Apply theme class to document root
  useEffect(() => {
    localStorage.setItem("theme", theme);
    const root = document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add(theme);
  }, [theme]);

  const sessionTrendData = sessions
    .map((session) => {
      const sessionTopFiles = session.messages
        .filter((message) => message.role === "assistant" && message.metadata?.top_files)
        .flatMap((message) => (message.metadata?.top_files as TopFileInfo[]) ?? []);

      if (sessionTopFiles.length === 0) return null;

      const avgBestScore =
        sessionTopFiles.reduce((sum, file) => sum + (file.best_score ?? 0), 0) /
        sessionTopFiles.length;
      const avgScore =
        sessionTopFiles.reduce((sum, file) => sum + (file.avg_score ?? file.best_score ?? 0), 0) /
        sessionTopFiles.length;
      const avgHits =
        sessionTopFiles.reduce((sum, file) => sum + (file.hits ?? 0), 0) /
        sessionTopFiles.length;

      return {
        sessionName: session.name,
        best_score: avgBestScore,
        avg_score: avgScore,
        hits: avgHits,
      };
    })
    .filter(
      (item): item is { sessionName: string; best_score: number; avg_score: number; hits: number } =>
        item !== null
    );

  const maxSessionTrendValue = Math.max(
    0.01,
    ...sessionTrendData.map((item) => item[trendMetric])
  );

  const makeSessionTitle = (questionStr: string) => {
    const trimmed = questionStr.trim();
    if (!trimmed) return "Cuộc trò chuyện mới";
    return trimmed.length > 32 ? `${trimmed.slice(0, 32)}...` : trimmed;
  };

  const isDefaultSessionTitle = (title: string) =>
    title.startsWith("Session ") || title === "Cuộc trò chuyện mới" || title === "Cuộc chat mới";

  const persistSessions = (updatedSessions: ChatSession[]) => {
    setSessions(updatedSessions);
    fetch(SESSIONS_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(updatedSessions),
    }).catch(() => {
      // Fail silently, keep local state
    });
  };

  const loadSession = (sessionId: string) => {
    const session = sessions.find((item) => item.id === sessionId);
    if (!session) return;
    setActiveSessionId(sessionId);
    setMessages(session.messages);
    setActiveTab("chat");
  };

  const createNewSession = () => {
    const name = `Session ${sessions.length + 1}`;
    const newSession: ChatSession = {
      id: `session-${Date.now()}`,
      name,
      createdAt: new Date().toLocaleString(),
      messages: DEFAULT_MESSAGES,
    };
    persistSessions([newSession, ...sessions]);
    setActiveSessionId(newSession.id);
    setMessages(newSession.messages);
    setChatDropdownOpen(true);
  };

  const deleteSession = (sessionId: string) => {
    const remaining = sessions.filter((session) => session.id !== sessionId);
    if (remaining.length === 0) {
      const newSession: ChatSession = {
        id: `session-${Date.now()}`,
        name: "Session 1",
        createdAt: new Date().toLocaleString(),
        messages: DEFAULT_MESSAGES,
      };
      persistSessions([newSession]);
      setActiveSessionId(newSession.id);
      setMessages(newSession.messages);
      return;
    }

    persistSessions(remaining);
    if (activeSessionId === sessionId) {
      setActiveSessionId(remaining[0].id);
      setMessages(remaining[0].messages);
    }
  };

  const updateCurrentSessionMessages = (updatedMessages: Message[], persist = true) => {
    setMessages(updatedMessages);
    const updatedSessions = sessions.map((session) => {
      if (session.id !== activeSessionId) return session;
      const updatedSession = { ...session, messages: updatedMessages };

      const hasOneGreetingMessage =
        session.messages.length === 1 && session.messages[0].role === "assistant";
      const firstUserMessage = updatedMessages.find((message) => message.role === "user");
      if (hasOneGreetingMessage && firstUserMessage && isDefaultSessionTitle(session.name)) {
        updatedSession.name = makeSessionTitle(firstUserMessage.content);
      }
      return updatedSession;
    });
    if (persist) {
      persistSessions(updatedSessions);
    } else {
      setSessions(updatedSessions);
    }
  };

  useEffect(() => {
    const loadSessionsFromServer = async () => {
      try {
        const response = await fetch(SESSIONS_API_URL);
        if (response.ok) {
          const parsed = (await response.json()) as ChatSession[];
          if (parsed.length > 0) {
            setSessions(parsed);
            setActiveSessionId(parsed[0].id);
            setMessages(parsed[0].messages);
            return;
          }
        }
      } catch {
        // ignore fetch errors
      }
      createNewSession();
    };

    void loadSessionsFromServer();
  }, []);

  const sendMessage = async (
    event?: FormEvent<HTMLFormElement> | MouseEvent<HTMLButtonElement>
  ) => {
    event?.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;

    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const userMessage: Message = { role: "user", content: trimmed, time: timestamp };
    const updatedMessages = [...messages, userMessage];
    updateCurrentSessionMessages(updatedMessages, false);
    setQuestion("");
    setLoading(true);

    try {
      const historyLimit = 10;
      const condensedHistory = updatedMessages.slice(-historyLimit).map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmed, history: condensedHistory, stream: true }),
      });

      if (!response.ok) {
        const errorData = (await response.json()) as { detail?: string };
        throw new Error(errorData.detail || "API error");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      let done = false;
      let assistantAnswer = "";
      let metadata: any = null;
      let buffer = "";

      const initialAssistantMessage: Message = {
        role: "assistant",
        content: "",
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setMessages([...updatedMessages, initialAssistantMessage]);

      while (!done && reader) {
        const { value, done: readDone } = await reader.read();
        done = readDone;
        if (value) {
          buffer += decoder.decode(value, { stream: !done });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (trimmedLine.startsWith("data: ")) {
              const dataStr = trimmedLine.slice(6).trim();
              if (dataStr === "[DONE]") {
                break;
              }
              try {
                const parsed = JSON.parse(dataStr);
                if (parsed.event === "metadata") {
                  metadata = parsed.data;
                } else if (parsed.event === "token") {
                  assistantAnswer += parsed.data;
                  setMessages([
                    ...updatedMessages,
                    {
                      role: "assistant",
                      content: assistantAnswer,
                      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                      metadata: metadata || undefined,
                    },
                  ]);
                }
              } catch (e) {
                // ignore JSON parse errors for incomplete packets
              }
            }
          }
        }
      }

      const answerTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const assistantMessage: Message = {
        role: "assistant",
        content: assistantAnswer,
        time: answerTime,
        metadata: metadata ? {
          rewritten_query: metadata.rewritten_query,
          used_rewrite: metadata.used_rewrite,
          show_rewritten_query: metadata.show_rewritten_query,
          grounded: metadata.grounded,
          warning: metadata.warning,
          mode: metadata.mode,
          top_files: metadata.top_files,
        } : undefined,
      };
      const updatedMessagesWithAnswer = [...updatedMessages, assistantMessage];
      updateCurrentSessionMessages(updatedMessagesWithAnswer, true);
    } catch (err) {
      const errorMessage = (err as Error).message;
      const errorMessageText = `Lỗi: ${errorMessage}`;
      const errorMessageEntry: Message = {
        role: "assistant",
        content: errorMessageText,
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      const updatedMessagesWithError = [...updatedMessages, errorMessageEntry];
      updateCurrentSessionMessages(updatedMessagesWithError, true);
    } finally {
      setLoading(false);
    }
  };

  const fetchEvaluation = async (forceRefresh = false) => {
    setEvaluationLoading(true);
    setEvaluationError("");

    try {
      const url = forceRefresh
        ? "http://127.0.0.1:8000/evaluation?force_refresh=true"
        : "http://127.0.0.1:8000/evaluation";
      const response = await fetch(url);
      if (!response.ok) {
        const errorData = (await response.json()) as { detail?: string };
        throw new Error(errorData.detail || "API error");
      }
      const data = (await response.json()) as EvaluationResponse;
      setEvaluationResults(data.results);
    } catch (err) {
      setEvaluationError((err as Error).message);
    } finally {
      setEvaluationLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === "evaluation" && evaluationResults.length === 0 && !evaluationLoading) {
      void fetchEvaluation(false);
    }
  }, [activeTab]);

  useEffect(() => {
    if (!sidebarOpen) {
      setChatDropdownOpen(false);
    }
  }, [sidebarOpen]);

  const activeSessionName =
    sessions.find((session) => session.id === activeSessionId)?.name || "Cuộc trò chuyện mới";

  return (
    <div className="app-shell">
      {/* Sidebar Component */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onLoadSession={loadSession}
        onCreateNewSession={createNewSession}
        onDeleteSession={deleteSession}
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        chatDropdownOpen={chatDropdownOpen}
        setChatDropdownOpen={setChatDropdownOpen}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
        theme={theme}
        setTheme={setTheme}
      />

      {/* Main Content Area */}
      <main className="content-area">
        {activeTab === "chat" && (
          <ChatScreen
            messages={messages}
            loading={loading}
            question={question}
            setQuestion={setQuestion}
            onSendMessage={sendMessage}
            activeSessionName={activeSessionName}
            onOpenMobileMenu={() => setMobileOpen(true)}
          />
        )}

        {activeTab === "evaluation" && (
          <EvaluationScreen
            results={evaluationResults}
            loading={evaluationLoading}
            error={evaluationError}
            sessionTrendData={sessionTrendData}
            trendMetric={trendMetric}
            setTrendMetric={setTrendMetric}
            maxSessionTrendValue={maxSessionTrendValue}
            onRefresh={() => void fetchEvaluation(true)}
            onOpenMobileMenu={() => setMobileOpen(true)}
          />
        )}

        {activeTab === "settings" && (
          <SettingsScreen onOpenMobileMenu={() => setMobileOpen(true)} />
        )}
      </main>
    </div>
  );
}

export default App;
