"use client";

import { useState, useEffect } from "react";
import { Message, ChatResponse, QueryTranslationType, RoutingType, QueryConstructionType, IndexTechnique, RetrievalType, GenerationType } from "../../types";
import { API_ENDPOINTS } from "../../constants/api";
import RadioGroup from "../common/RadioGroup";

interface ChatTabProps {
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

export default function ChatTab({ messages, setMessages }: ChatTabProps) {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedQueryTranslation, setSelectedQueryTranslation] = useState<QueryTranslationType>("none");
  const [selectedRoutingType, setSelectedRoutingType] = useState<RoutingType>("logical");
  const [selectedQueryConstruction, setSelectedQueryConstruction] = useState<QueryConstructionType>("none");
  const [selectedIndexing, setSelectedIndexing] = useState<IndexTechnique>("default");
  const [selectedRetrieval, setSelectedRetrieval] = useState<RetrievalType>("none");
  const [selectedGeneration, setSelectedGeneration] = useState<GenerationType>("none");

  // Reset selections when they become disabled
  useEffect(() => {
    if (selectedQueryTranslation === "decomposition") {
      // Reset routing and query construction when decomposition is selected
      setSelectedRoutingType("none");
      setSelectedQueryConstruction("none");
    }
  }, [selectedQueryTranslation]);

  useEffect(() => {
    if (selectedRoutingType !== "logical" && selectedIndexing === "multi-representation") {
      // Reset indexing to default if multi-representation becomes disabled
      setSelectedIndexing("default");
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

          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              // Stream ended, save the accumulated message
              if (accumulatedContent) {
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
                      setMessages(prev => [...prev, {
                        role: 'assistant',
                        content: accumulatedContent,
                        sources: sources.length > 0 ? sources : undefined
                      }]);
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


  // Conditional disabling logic
  const isDecompositionSelected = selectedQueryTranslation === "decomposition";
  const isLogicalRoutingSelected = selectedRoutingType === "logical";

  const queryTranslationOptions = [
    // Query Translation
    { label: "None", value: "none" as QueryTranslationType },
    { label: "Multi-Query", value: "multi-query" as QueryTranslationType },
    { label: "RAG Fusion", value: "rag-fusion" as QueryTranslationType },
    { label: "Decomposition", value: "decomposition" as QueryTranslationType },
    { label: "Step Back", value: "step-back" as QueryTranslationType },
    { label: "HyDE", value: "hyde" as QueryTranslationType },
  ];

  const routingOptions = [
    { label: "None", value: "none" as RoutingType, disabled: isDecompositionSelected },
    { label: "Logical", value: "logical" as RoutingType, disabled: isDecompositionSelected },
    { label: "Semantic", value: "semantic" as RoutingType, disabled: true },
  ];

  const queryConstructionOptions = [
    { label: "None", value: "none" as QueryConstructionType, disabled: isDecompositionSelected },
    { label: "Vector", value: "vector" as QueryConstructionType, disabled: isDecompositionSelected },
    { label: "SQL", value: "sql" as QueryConstructionType, disabled: true }
  ]

  const indexingOptions = [
    { label: "Default", value: "default" as IndexTechnique },
    { label: "Multi-Representation", value: "multi-representation" as IndexTechnique },
    { label: "RAPTOR", value: "raptor" as IndexTechnique }, // Always disabled for now
    { label: "ColBERT", value: "colbert" as IndexTechnique } // Always disabled for now
  ];

  const retrievalOptions = [
    { label: "None", value: "none" as RetrievalType, disabled: false },
    { label: "CRAG", value: "crag" as RetrievalType, disabled: false },
  ]

  return (
    <div className="flex flex-col h-[600px]">
      <div className="flex gap-4 mb-4">
        <RadioGroup
          title="Query Translation"
          name="query-translation"
          options={queryTranslationOptions}
          selectedValue={selectedQueryTranslation}
          onChange={setSelectedQueryTranslation}
        />
        <RadioGroup
          title="Routing"
          name="routing"
          options={routingOptions}
          selectedValue={selectedRoutingType}
          onChange={setSelectedRoutingType}
        />
        <RadioGroup
          title="Query Construction"
          name="query-construction"
          options={queryConstructionOptions}
          selectedValue={selectedQueryConstruction}
          onChange={setSelectedQueryConstruction}
        />
        <RadioGroup
          title="Indexing"
          name="indexing"
          options={indexingOptions}
          selectedValue={selectedIndexing}
          onChange={setSelectedIndexing}
        />
        <RadioGroup
          title="Retrieval"
          name="retrieval"
          options={retrievalOptions}
          selectedValue={selectedRetrieval}
          onChange={setSelectedRetrieval}
        />
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