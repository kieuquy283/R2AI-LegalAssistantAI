export type MessageMetadata = {
  rewritten_query?: string;
  used_rewrite?: boolean;
  show_rewritten_query?: boolean;
  grounded?: boolean;
  warning?: string;
  mode?: string;
  top_files?: TopFileInfo[];
};

export type Message = {
  role: "user" | "assistant";
  content: string;
  time?: string;
  metadata?: MessageMetadata;
};

export type TopFileInfo = {
  source_file: string;
  best_score: number;
  avg_score?: number;
  hits: number;
};

export type ChatResponse = {
  answer: string;
  rewritten_query: string;
  used_rewrite: boolean;
  show_rewritten_query: boolean;
  grounded: boolean;
  warning: string;
  mode: string;
  top_files: TopFileInfo[];
  history: Message[];
};

export type EvaluationStats = {
  name: string;
  top_k: number;
  eval_path: string;
  sample_count: number;
  hit: number;
  recall: number;
  mrr: number;
};

export type EvaluationResponse = {
  results: EvaluationStats[];
};

export type ChatSession = {
  id: string;
  name: string;
  createdAt: string;
  messages: Message[];
};

export type TabKey = "chat" | "evaluation" | "settings";
