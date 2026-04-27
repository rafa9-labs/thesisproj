import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL ?? "/api/v1";

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 502 || error.response?.status === 503) {
      console.warn("[API] Backend unavailable");
    }
    return Promise.reject(error);
  },
);

export default apiClient;
