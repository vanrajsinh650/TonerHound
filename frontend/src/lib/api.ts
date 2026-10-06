import axios from "axios";
import type { VerifyResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: API_URL,
  timeout: 60_000,
});

export async function verifyDocument(
  file: File,
  extraction: string
): Promise<VerifyResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("extraction", extraction);

  const response = await client.post<VerifyResponse>("/api/verify", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
}
