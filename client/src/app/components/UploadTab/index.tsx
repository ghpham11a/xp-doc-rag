"use client";

import { useState } from "react";
import { DropZoneId, IndexTechnique } from "../../types";
import { FILE_UPLOAD_CONFIG } from "../../constants/api";
import { useFileUpload } from "../../hooks/useFileUpload";
import RadioGroup from "../common/RadioGroup";

export default function UploadTab() {
  const [dragActive, setDragActive] = useState<{ [key: string]: boolean }>({ subject_one: false, subject_two: false });
  const { uploading, uploadedFiles, uploadFiles, clearVectorStore } = useFileUpload();
  const [selectedOption, setSelectedOption] = useState<IndexTechnique>("default");

  const handleDrag = (e: React.DragEvent, zone: DropZoneId) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(prev => ({ ...prev, [zone]: true }));
    } else if (e.type === "dragleave") {
      setDragActive(prev => ({ ...prev, [zone]: false }));
    }
  };

  const handleDrop = (e: React.DragEvent, zone: DropZoneId, selectedOption: IndexTechnique) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(prev => ({ ...prev, [zone]: false }));

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFiles(e.dataTransfer.files, zone, selectedOption);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>, zone: DropZoneId, selectedOption: IndexTechnique) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      uploadFiles(e.target.files, zone, selectedOption);
    }
  };

  const conceptOptions = [
    { label: "Default", value: "default" as IndexTechnique },
    { label: "Multi-Representation", value: "multi-representation" as IndexTechnique },
    { label: "RAPTOR", value: "raptor" as IndexTechnique },
    { label: "ColBERT", value: "colbert" as IndexTechnique }
  ];


  const renderDropZone = (zone: DropZoneId, title: string) => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-700">{title}</h3>
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center ${dragActive[zone] ? "border-blue-500 bg-blue-50" : "border-gray-300"
          }`}
        onDragEnter={(e) => handleDrag(e, zone)}
        onDragLeave={(e) => handleDrag(e, zone)}
        onDragOver={(e) => handleDrag(e, zone)}
        onDrop={(e) => handleDrop(e, zone, selectedOption)}
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
              onChange={(e) => handleChange(e, zone, selectedOption)}
              multiple
              accept={FILE_UPLOAD_CONFIG.ACCEPTED_FORMATS}
              disabled={uploading}
            />
          </label>
          <span> or drag and drop</span>
        </p>
        <p className="text-xs text-gray-500 mt-1">
          PDF, DOC, DOCX, TXT up to {FILE_UPLOAD_CONFIG.MAX_SIZE_MB}MB
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
      <div className="flex gap-4 mb-4">
        <RadioGroup
          title="Indexing"
          name="indexing"
          options={conceptOptions}
          selectedValue={selectedOption}
          onChange={setSelectedOption}
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {renderDropZone('subject_one', 'Subject 1')}
        {renderDropZone('subject_two', 'Subject 2')}
      </div>
    </div>
  );
}