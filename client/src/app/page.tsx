"use client";

import { useState } from "react";
//

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  technique?: string;
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<"upload" | "chat">("upload");
  const [messages, setMessages] = useState<Message[]>([]);

  return (
    <div className="min-h-screen p-8">
      <main className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Document Chat</h1>

        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab("upload")}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${activeTab === "upload"
                ? "border-blue-500 text-blue-600"
                : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
            >
              Upload Documents
            </button>
            <button
              onClick={() => setActiveTab("chat")}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${activeTab === "chat"
                ? "border-blue-500 text-blue-600"
                : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
            >
              Chat
            </button>
          </nav>
        </div>

        <div className="mt-8">
          {activeTab === "upload" ? <UploadTab /> : <ChatTab messages={messages} setMessages={setMessages} />}
        </div>
      </main>
    </div>
  );
}

function UploadTab() {
  const [dragActive, setDragActive] = useState<{ [key: string]: boolean }>({ zone1: false, zone2: false, zone3: false });
  const [uploading, setUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<{
    zone1: Array<{ filename: string, saved_as: string, size: number }>,
    zone2: Array<{ filename: string, saved_as: string, size: number }>
  }>({ zone1: [], zone2: [] });

  const uploadFiles = async (files: FileList, zone: 'zone1' | 'zone2') => {
    setUploading(true);
    const formData = new FormData();

    if (files.length === 1) {
      formData.append("file", files[0]);

      try {
        const response = await fetch("http://localhost:8000/files/upload", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          console.log("File uploaded successfully:", data);
          setUploadedFiles(prev => ({ ...prev, [zone]: [...prev[zone], data] }));
        } else {
          console.error("Upload failed:", response.statusText);
        }
      } catch (error) {
        console.error("Upload error:", error);
      }
    } else {
      for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
      }

      try {
        const response = await fetch("http://localhost:8000/files/upload-multiple", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          setUploadedFiles(prev => ({ ...prev, [zone]: [...prev[zone], ...data.uploaded_files.filter((f: any) => !f.error)] }));
        } else {
          console.error("Upload failed:", response.statusText);
        }
      } catch (error) {
        console.error("Upload error:", error);
      }
    }

    setUploading(false);
  };

  const handleDrag = (e: React.DragEvent, zone: 'zone1' | 'zone2') => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(prev => ({ ...prev, [zone]: true }));
    } else if (e.type === "dragleave") {
      setDragActive(prev => ({ ...prev, [zone]: false }));
    }
  };

  const handleDrop = (e: React.DragEvent, zone: 'zone1' | 'zone2') => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(prev => ({ ...prev, [zone]: false }));

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFiles(e.dataTransfer.files, zone);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>, zone: 'zone1' | 'zone2') => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      uploadFiles(e.target.files, zone);
    }
  };

  const clearVectorStore = async (zone: 'zone1' | 'zone2') => {
    try {
      const response = await fetch("http://localhost:8000/files/clear-vector-store", {
        method: "DELETE",
      });

      if (response.ok) {
        const data = await response.json();
        console.log(`Vector store cleared for ${zone}:`, data);
        // Clear the uploaded files for this zone
        setUploadedFiles(prev => ({ ...prev, [zone]: [] }));
      } else {
        console.error("Failed to clear vector store:", response.statusText);
      }
    } catch (error) {
      console.error("Error clearing vector store:", error);
    }
  };

  const renderDropZone = (zone: 'zone1' | 'zone2', title: string) => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-700">{title}</h3>
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center ${dragActive[zone] ? "border-blue-500 bg-blue-50" : "border-gray-300"
          }`}
        onDragEnter={(e) => handleDrag(e, zone)}
        onDragLeave={(e) => handleDrag(e, zone)}
        onDragOver={(e) => handleDrag(e, zone)}
        onDrop={(e) => handleDrop(e, zone)}
      >
        <svg
          className="mx-auto h-10 w-10 text-gray-400"
          stroke="currentColor"
          fill="none"
          viewBox="0 0 48 48"
        >
          <path
            d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <p className="mt-2 text-sm text-gray-600">
          <label htmlFor={`file-upload-${zone}`} className="cursor-pointer text-blue-600 hover:text-blue-500">
            <span>Upload a file</span>
            <input
              id={`file-upload-${zone}`}
              name={`file-upload-${zone}`}
              type="file"
              className="sr-only"
              onChange={(e) => handleChange(e, zone)}
              multiple
              accept=".pdf,.doc,.docx,.txt"
              disabled={uploading}
            />
          </label>
          <span> or drag and drop</span>
        </p>
        <p className="text-xs text-gray-500 mt-1">
          PDF, DOC, DOCX, TXT up to 10MB
        </p>
      </div>

      <div className="bg-gray-50 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm font-medium text-gray-900">Uploaded Files</h4>
          <button
            onClick={() => clearVectorStore(zone)}
            className="px-3 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500"
          >
            Clear Vector Store
          </button>
        </div>
        {uploading && (
          <p className="text-sm text-blue-600 mb-2">Uploading files...</p>
        )}
        {uploadedFiles[zone].length === 0 ? (
          <p className="text-sm text-gray-500">No files uploaded yet</p>
        ) : (
          <ul className="space-y-2">
            {uploadedFiles[zone].map((file, index) => (
              <li key={index} className="flex items-center justify-between text-sm">
                <span className="text-gray-700">{file.filename}</span>
                <span className="text-gray-500">{(file.size / 1024).toFixed(1)} KB</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );

  return (
    <div className="space-y-8">
      {renderDropZone('zone1', 'Drop Zone 1')}
      {renderDropZone('zone2', 'Drop Zone 2')}
    </div>
  );
}

interface ChatTabProps {
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

function ChatTab({ messages, setMessages }: ChatTabProps) {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !loading) {
      const userMessage = message.trim();
      setMessage("");
      setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
      setLoading(true);

      try {
        const response = await fetch("http://localhost:8000/chats/send", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message: userMessage, technique: selectedOption }),
        });

        if (response.ok) {
          const data = await response.json();
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
      } catch (error) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'Error: Failed to connect to the server'
        }]);
      } finally {
        setLoading(false);
      }
    }
  };

  const [selectedOption, setSelectedOption] = useState("");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedOption(e.target.value);
  };

  return (
    <div className="flex flex-col h-[600px]">

      <div className="flex gap-4 mb-4">
        <form className="bg-white p-4 rounded-lg shadow flex-1">
          <h2 className="text-xl font-bold text-gray-800">Query Translation</h2>

          {[
            "None",
            "Multi-Query",
            "RAG Fusion",
            "Decomposition",
            "Step Back",
            "HyDE",
          ].map((label) => {
            const value = label.toLowerCase().replace(/\s+/g, "-");
            return (
              <label
                key={value}
                className="flex items-center space-x-3 cursor-pointer"
              >
                <input
                  type="radio"
                  name="queryTranslation"
                  value={value}
                  checked={selectedOption === value}
                  onChange={handleChange}
                  className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                />
                <span className="text-gray-700">{label}</span>
              </label>
            );
          })}
        </form>

        <form className="bg-white p-4 rounded-lg shadow flex-1">
          <h2 className="text-xl font-bold text-gray-800">Routing</h2>

          {[
            "None",
            "Logical",
            "Semantic",
          ].map((label) => {
            const value = label.toLowerCase().replace(/\s+/g, "-");
            return (
              <label
                key={value}
                className="flex items-center space-x-3 cursor-pointer"
              >
                <input
                  type="radio"
                  name="queryTranslation"
                  value={value}
                  checked={selectedOption === value}
                  onChange={handleChange}
                  className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                />
                <span className="text-gray-700">{label}</span>
              </label>
            );
          })}
        </form>

        <form className="bg-white p-4 rounded-lg shadow flex-1">
          <h2 className="text-xl font-bold text-gray-800">Query Construction</h2>

          {[
            "None",
            "Structured",
          ].map((label) => {
            const value = label.toLowerCase().replace(/\s+/g, "-");
            return (
              <label
                key={value}
                className="flex items-center space-x-3 cursor-pointer"
              >
                <input
                  type="radio"
                  name="queryTranslation"
                  value={value}
                  checked={selectedOption === value}
                  onChange={handleChange}
                  className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                />
                <span className="text-gray-700">{label}</span>
              </label>
            );
          })}
        </form>

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
                  className={`max-w-[80%] rounded-lg p-3 ${msg.role === 'user'
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
            {loading && (
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
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </form>
    </div>
  );
}