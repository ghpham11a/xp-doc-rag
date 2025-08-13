"use client";

import { useState, useEffect } from "react";
import RadioGroup from "./RadioGroup";
import { 
  QueryTranslationType, 
  RoutingType, 
  QueryConstructionType, 
  IndexTechnique, 
  RetrievalType, 
  GenerationType 
} from "../../types";

interface QueryTechniqueModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedQueryTranslation: QueryTranslationType;
  setSelectedQueryTranslation: (value: QueryTranslationType) => void;
  selectedRoutingType: RoutingType;
  setSelectedRoutingType: (value: RoutingType) => void;
  selectedQueryConstruction: QueryConstructionType;
  setSelectedQueryConstruction: (value: QueryConstructionType) => void;
  selectedIndexing: IndexTechnique;
  setSelectedIndexing: (value: IndexTechnique) => void;
  selectedRetrieval: RetrievalType;
  setSelectedRetrieval: (value: RetrievalType) => void;
  selectedGeneration: GenerationType;
  setSelectedGeneration: (value: GenerationType) => void;
}

export default function QueryTechniqueModal({
  isOpen,
  onClose,
  selectedQueryTranslation,
  setSelectedQueryTranslation,
  selectedRoutingType,
  setSelectedRoutingType,
  selectedQueryConstruction,
  setSelectedQueryConstruction,
  selectedIndexing,
  setSelectedIndexing,
  selectedRetrieval,
  setSelectedRetrieval,
  selectedGeneration,
  setSelectedGeneration,
}: QueryTechniqueModalProps) {
  if (!isOpen) return null;

  const queryTranslationOptions = [
    { label: "None", value: "none" as QueryTranslationType },
    { label: "Multi-Query", value: "multi-query" as QueryTranslationType },
    { label: "RAG Fusion", value: "rag-fusion" as QueryTranslationType },
    { label: "Recursive Decomposition", value: "recursive-decomposition" as QueryTranslationType },
    { label: "Individual Decomposition", value: "individual-decomposition" as QueryTranslationType },
    { label: "Step Back", value: "step-back" as QueryTranslationType },
    { label: "HyDE", value: "hyde" as QueryTranslationType },
  ];

  const routingOptions = [
    { label: "None", value: "none" as RoutingType, disabled: false },
    { label: "Logical", value: "logical" as RoutingType, disabled: false },
    { label: "Semantic", value: "semantic" as RoutingType, disabled: true },
  ];

  const queryConstructionOptions = [
    { label: "None", value: "none" as QueryConstructionType, disabled: false },
    { label: "Vector", value: "vector" as QueryConstructionType, disabled: false },
    { label: "SQL", value: "sql" as QueryConstructionType, disabled: true }
  ];

  const indexingOptions = [
    { label: "Default", value: "default" as IndexTechnique },
    { label: "Multi-Representation", value: "multi-representation" as IndexTechnique },
    { label: "RAPTOR", value: "raptor" as IndexTechnique },
    { label: "ColBERT", value: "colbert" as IndexTechnique }
  ];

  const retrievalOptions = [
    { label: "None", value: "none" as RetrievalType, disabled: false },
    { label: "CRAG", value: "crag" as RetrievalType, disabled: false },
  ];

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 z-40"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900">
              Query Technique Settings
            </h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-lg p-1"
            >
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            <div className="grid grid-cols-2 gap-6">
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
          </div>

          {/* Footer */}
          <div className="flex justify-end gap-3 px-6 py-4 border-t border-gray-200">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500"
            >
              Cancel
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              Apply Settings
            </button>
          </div>
        </div>
      </div>
    </>
  );
}