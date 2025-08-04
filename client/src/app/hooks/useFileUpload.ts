import { useState } from "react";
import { DropZoneId, UploadedFile, UploadResponse, MultipleUploadResponse, IndexTechnique } from "../types";
import { API_ENDPOINTS } from "../constants/api";

export function useFileUpload() {
  const [uploading, setUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<{
    subject_one: UploadedFile[];
    subject_two: UploadedFile[];
  }>({ subject_one: [], subject_two: [] });

  const uploadFiles = async (files: FileList, zone: DropZoneId, path: IndexTechnique) => {
    setUploading(true);
    
    const formData = new FormData();
    formData.append("zone", zone);
    formData.append("path", path);

    try {
      if (files.length === 1) {
        formData.append("file", files[0]);

        const response = await fetch(API_ENDPOINTS.UPLOAD_SINGLE, {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data: UploadResponse = await response.json();
          console.log("File uploaded successfully:", data);
          setUploadedFiles(prev => ({ 
            ...prev, 
            [zone]: [...prev[zone], data] 
          }));
          return { success: true, data };
        } else {
          console.error("Upload failed:", response.statusText);
          return { success: false, error: response.statusText };
        }
      } else {
        for (let i = 0; i < files.length; i++) {
          formData.append("files", files[i]);
        }

        const response = await fetch(API_ENDPOINTS.UPLOAD_MULTIPLE, {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data: MultipleUploadResponse = await response.json();
          const successfulFiles = data.uploaded_files.filter(f => !f.error);
          setUploadedFiles(prev => ({ 
            ...prev, 
            [zone]: [...prev[zone], ...successfulFiles] 
          }));
          return { success: true, data };
        } else {
          console.error("Upload failed:", response.statusText);
          return { success: false, error: response.statusText };
        }
      }
    } catch (error) {
      console.error("Upload error:", error);
      return { success: false, error: error instanceof Error ? error.message : "Unknown error" };
    } finally {
      setUploading(false);
    }
  };

  const clearVectorStore = async (zone: DropZoneId) => {
    try {
      const response = await fetch(API_ENDPOINTS.CLEAR_VECTOR_STORE, {
        method: "DELETE",
      });

      if (response.ok) {
        const data = await response.json();
        console.log(`Vector store cleared for ${zone}:`, data);
        setUploadedFiles(prev => ({ ...prev, [zone]: [] }));
        return { success: true, data };
      } else {
        console.error("Failed to clear vector store:", response.statusText);
        return { success: false, error: response.statusText };
      }
    } catch (error) {
      console.error("Error clearing vector store:", error);
      return { success: false, error: error instanceof Error ? error.message : "Unknown error" };
    }
  };

  return {
    uploading,
    uploadedFiles,
    uploadFiles,
    clearVectorStore,
  };
}