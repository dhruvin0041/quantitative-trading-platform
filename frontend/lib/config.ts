export const API_KEY = "hydra-secure-key-2026";

export const getBaseUrl = () => {
  if (typeof window !== 'undefined') {
    return `http://${window.location.hostname}:8000`;
  }
  return "http://127.0.0.1:8000";
};
