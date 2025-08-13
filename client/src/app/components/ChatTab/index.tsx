"use client";

import { useState, useEffect } from "react";
import { Message, ChatResponse, QueryTranslationType, RoutingType, QueryConstructionType, IndexTechnique, RetrievalType, GenerationType } from "../../types";
import { API_ENDPOINTS } from "../../constants/api";
import QueryTechniqueModal from "../common/QueryTechniqueModal";

interface ChatTabProps {
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

export default function ChatTab({ messages, setMessages }: ChatTabProps) {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedQueryTranslation, setSelectedQueryTranslation] = useState<QueryTranslationType>("none");
  const [selectedRoutingType, setSelectedRoutingType] = useState<RoutingType>("logical");
  const [selectedQueryConstruction, setSelectedQueryConstruction] = useState<QueryConstructionType>("none");
  const [selectedIndexing, setSelectedIndexing] = useState<IndexTechnique>("default");
  const [selectedRetrieval, setSelectedRetrieval] = useState<RetrievalType>("none");
  const [selectedGeneration, setSelectedGeneration] = useState<GenerationType>("none");

  // Reset selections when they become disabled
  useEffect(() => {
    if (selectedQueryTranslation === "recursive-decomposition") {
      // Reset routing and query construction when decomposition is selected
      // setSelectedRoutingType("none");
      // setSelectedQueryConstruction("none");
    }
  }, [selectedQueryTranslation]);

  useEffect(() => {
    if (selectedRoutingType !== "logical" && selectedIndexing === "multi-representation") {
      // Reset indexing to default if multi-representation becomes disabled
      // setSelectedIndexing("default");
    }
  }, [selectedRoutingType, selectedIndexing]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !loading && !isStreaming) {
      const userMessage = message.trim();
      setMessage("");
      setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
      setLoading(true);
      setIsStreaming(true);
      setStreamingMessage("");

      try {
        const response = await fetch('http://localhost:8000/chats/send-stream', {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ 
            message: userMessage, 
            path: [selectedQueryTranslation, selectedRoutingType, selectedQueryConstruction, selectedIndexing, selectedRetrieval, selectedGeneration]
          }),
        });

        if (response.ok && response.body) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let accumulatedContent = "";
          let sources: string[] = [];

          let messageAdded = false;
          
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              // Stream ended, save the accumulated message only if we haven't already
              if (accumulatedContent && !messageAdded) {
                setMessages(prev => [...prev, {
                  role: 'assistant',
                  content: accumulatedContent,
                  sources: sources.length > 0 ? sources : undefined
                }]);
                setStreamingMessage("");
                setIsStreaming(false);
              }
              break;
            }

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            console.log(lines)

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const jsonStr = line.slice(6);
                if (jsonStr.trim()) {
                  try {
                    const data = JSON.parse(jsonStr);
                    
                    if (data.type === 'content') {
                      // Skip null content
                      if (data.content === null || data.content === undefined) {
                        continue;
                      }
                      
                      // Handle the new format: {'generate': {'messages': ['content']}}
                      if (data.content && typeof data.content === 'object') {
                        // Extract message from the nested structure
                        if (data.content.generate?.messages?.[0]) {
                          accumulatedContent += data.content.generate.messages[0];
                        } else if (data.content.messages?.[0]) {
                          accumulatedContent += data.content.messages[0];
                        } else {
                          // Fallback to treating content as string if it's not in expected format
                          accumulatedContent += JSON.stringify(data.content);
                        }
                      } else {
                        accumulatedContent += data.content;
                      }
                      setStreamingMessage(accumulatedContent);
                    } else if (data.type === 'sources') {
                      sources = data.sources;
                    } else if (data.type === 'done') {
                      // Streaming complete
                      if (accumulatedContent && !messageAdded) {
                        setMessages(prev => [...prev, {
                          role: 'assistant',
                          content: accumulatedContent,
                          sources: sources.length > 0 ? sources : undefined
                        }]);
                        messageAdded = true;
                      }
                      setStreamingMessage("");
                      setIsStreaming(false);
                    } else if (data.type === 'error') {
                      setMessages(prev => [...prev, {
                        role: 'assistant',
                        content: `Error: ${data.error}`
                      }]);
                      setStreamingMessage("");
                      setIsStreaming(false);
                    }
                  } catch (e) {
                    console.error('Failed to parse SSE data:', e);
                  }
                }
              }
            }
          }
        } else {
          // Fallback to non-streaming endpoint if streaming fails
          const response = await fetch(API_ENDPOINTS.SEND_CHAT, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ 
              message: userMessage, 
              path: [selectedQueryTranslation, selectedRoutingType, selectedQueryConstruction, selectedIndexing, selectedRetrieval, selectedGeneration]
            }),
          });

          if (response.ok) {
            const data: ChatResponse = await response.json();
            setMessages(prev => [...prev, {
              role: 'assistant',
              content: data.answer,
              sources: data.sources
            }]);
          } else {
            const error = await response.json();
            setMessages(prev => [...prev, {
              role: 'assistant',
              content: `Error: ${error.detail || 'Failed to get response'}`
            }]);
          }
          setIsStreaming(false);
        }
      } catch (error) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'Error: Failed to connect to the server'
        }]);
        setStreamingMessage("");
        setIsStreaming(false);
      } finally {
        setLoading(false);
      }
    }
  };


  const getSettingsSummary = () => {
    const settings = [];
    if (selectedQueryTranslation !== "none") settings.push(selectedQueryTranslation);
    if (selectedRoutingType !== "none") settings.push(selectedRoutingType);
    if (selectedQueryConstruction !== "none") settings.push(selectedQueryConstruction);
    if (selectedIndexing !== "default") settings.push(selectedIndexing);
    if (selectedRetrieval !== "none") settings.push(selectedRetrieval);
    
    if (settings.length === 0) return "Default settings";
    return settings.join(", ");
  }

  return (
    <div className="flex flex-col h-[600px]">
      <QueryTechniqueModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        selectedQueryTranslation={selectedQueryTranslation}
        setSelectedQueryTranslation={setSelectedQueryTranslation}
        selectedRoutingType={selectedRoutingType}
        setSelectedRoutingType={setSelectedRoutingType}
        selectedQueryConstruction={selectedQueryConstruction}
        setSelectedQueryConstruction={setSelectedQueryConstruction}
        selectedIndexing={selectedIndexing}
        setSelectedIndexing={setSelectedIndexing}
        selectedRetrieval={selectedRetrieval}
        setSelectedRetrieval={setSelectedRetrieval}
        selectedGeneration={selectedGeneration}
        setSelectedGeneration={setSelectedGeneration}
      />
      
      <div className="flex items-center justify-between mb-4">
        <button
          onClick={() => setIsModalOpen(true)}
          className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
          </svg>
          <span>Query Settings</span>
        </button>
        <div className="text-sm text-gray-600">
          Active: {getSettingsSummary()}
        </div>
      </div>

      <div className="flex-1 bg-gray-50 rounded-lg p-4 mb-4 overflow-y-auto">
        {messages.length === 0 ? (
          <p className="text-gray-500 text-center mt-8">
            No messages yet. Upload documents and start asking questions!
          </p>
        ) : (
          <div className="space-y-4">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-white border border-gray-200'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-gray-200">
                      <p className="text-xs text-gray-500">Sources:</p>
                      <ul className="text-xs text-gray-600">
                        {msg.sources.map((source, idx) => (
                          <li key={idx}>• {source}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isStreaming && streamingMessage && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 rounded-lg p-3 max-w-[80%]">
                  <p className="whitespace-pre-wrap">
                    {streamingMessage}
                    <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse ml-1" />
                  </p>
                </div>
              </div>
            )}
            {loading && !isStreaming && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 rounded-lg p-3">
                  <p className="text-gray-500">Thinking...</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask about your documents..."
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading || isStreaming}
        />
        <button
          type="submit"
          disabled={loading || isStreaming}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </form>
    </div>
  );
}