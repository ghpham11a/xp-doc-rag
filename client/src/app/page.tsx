"use client";

import { useState } from "react";
import { Message, TabType } from "./types";
import TabNavigation from "./components/common/TabNavigation";
import UploadTab from "./components/UploadTab";
import ChatTab from "./components/ChatTab";

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>("upload");
  const [messages, setMessages] = useState<Message[]>([]);

  const tabs = [
    { key: "upload" as TabType, label: "Upload Documents" },
    { key: "chat" as TabType, label: "Chat" },
  ];

  return (
    <div className="min-h-screen p-8">
      <main className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Document Chat</h1>

        <TabNavigation
          activeTab={activeTab}
          onTabChange={setActiveTab}
          tabs={tabs}
        />

        <div className="mt-8">
          {activeTab === "upload" ? (
            <UploadTab />
          ) : (
            <ChatTab messages={messages} setMessages={setMessages} />
          )}
        </div>
      </main>
    </div>
  );
}