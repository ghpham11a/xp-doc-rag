export const API_BASE_URL = 'http://localhost:8000';

export const API_ENDPOINTS = {
  UPLOAD_SINGLE: `${API_BASE_URL}/files/upload`,
  UPLOAD_MULTIPLE: `${API_BASE_URL}/files/upload-multiple`,
  CLEAR_VECTOR_STORE: `${API_BASE_URL}/files/clear-vector-store`,
  SEND_CHAT: `${API_BASE_URL}/chats/send`,
} as const;

export const FILE_UPLOAD_CONFIG = {
  MAX_SIZE_MB: 10,
  ACCEPTED_FORMATS: '.pdf,.doc,.docx,.txt',
  ACCEPTED_MIME_TYPES: [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain'
  ],
} as const;