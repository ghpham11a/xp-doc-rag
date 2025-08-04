import { useState } from "react";
import { Message, ChatResponse, QueryTranslationType, RoutingType } from "../types";
import { API_ENDPOINTS } from "../constants/api";
import { Robots } from "next/dist/lib/metadata/types/metadata-types";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async (
    message: string, 
    path: (QueryTranslationType | RoutingType)[] = [],
  ): Promise<{ success: boolean; error?: string }> => {
    if (!message.trim()) {
      return { success: false, error: "Message cannot be empty" };
    }

    setMessages(prev => [...prev, { role: 'user', content: message }]);
    setLoading(true);

    try {
      const response = await fetch(API_ENDPOINTS.SEND_CHAT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message, path }),
      });

      if (response.ok) {
        const data: ChatResponse = await response.json();
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
          path: path
        }]);
        return { success: true };
      } else {
        const error = await response.json();
        const errorMessage = error.detail || 'Failed to get response';
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Error: ${errorMessage}`
        }]);
        return { success: false, error: errorMessage };
      }
    } catch (error) {
      const errorMessage = 'Failed to connect to the server';
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${errorMessage}`
      }]);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  };

  const clearMessages = () => {
    setMessages([]);
  };

  return {
    messages,
    loading,
    sendMessage,
    clearMessages,
    setMessages,
  };
}