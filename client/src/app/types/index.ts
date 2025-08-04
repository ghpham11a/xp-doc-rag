export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  path?: string[];
}

export interface UploadedFile {
  filename: string;
  saved_as: string;
  size: number;
}

export interface UploadResponse {
  filename: string;
  saved_as: string;
  size: number;
}

export interface MultipleUploadResponse {
  uploaded_files: Array<UploadedFile & { error?: string }>;
}

export interface ChatResponse {
  answer: string;
  sources?: string[];
}

export type TabType = "upload" | "chat";
export type DropZoneId = 'subject_one' | 'subject_two';
export type QueryTranslationType = 'none' | 'multi-query' | 'rag-fusion' | 'decomposition' | 'step-back' | 'hyde' | 'logical' | 'semantic' | 'structured';
export type RoutingType = 'none' | 'logical' | 'semantic';
export type QueryConstructionType = 'none' | 'vector'
export type IndexTechnique = 'default' | 'multi-representation' | 'raptor' | 'colbert';