export const API_KEY = "hydra-secure-key-2026";

export const getBaseUrl = () => {
  // 1. Priority: Explicit environment variable
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "");
  }

  // 2. Client-side dynamic resolution
  if (typeof window !== 'undefined') {
    const { hostname, protocol } = window.location;
    // Handle cases where we might be on a custom domain but backend is on 8000
    return `${protocol}//${hostname}:8000`;
  }

  // 3. Server-side default
  return "http://127.0.0.1:8000";
};
