# HYDRA TERMINAL — SECURITY HARDENING REPORT

### VULNERABILITY SUMMARY

| ID | Severity | File | Location | Finding |
| :--- | :--- | :--- | :--- | :--- |
| **SEC-01** | **CRITICAL** | `backend/api.py` | L76 | Hardcoded fallback `API_KEY` ("dev-secret-key-1234"). |
| **SEC-02** | **CRITICAL** | `backend/scripts/paper_trading_loop.py` | L9 | Hardcoded `API_KEY` ("dev-secret-key-1234"). |
| **SEC-03** | **HIGH** | `frontend/app/page.tsx` | L33, L59 | Hardcoded `X-API-Key` header ("dev-secret-key-1234"). |
| **SEC-04** | **HIGH** | `frontend/app/performance/page.tsx` | L16 | Hardcoded `X-API-Key` header ("dev-secret-key-1234"). |
| **SEC-05** | **HIGH** | `backend/src/data_ingestion/nlp_processor.py` | L15 | Gemini analyzer defaults to `GOOGLE_API_KEY` or `None`. |
| **SEC-06** | **MEDIUM** | `backend/api.py` | L97 | CORS `FRONTEND_URL` defaults to `localhost:3000`. |
| **SEC-07** | **MEDIUM** | `backend/api.py` | L127 | Global exception handler leaks class names (`type(exc).__name__`). |

***

### REMEDIATION PLAN

1.  **Strict Environment Enforcement**: Replace all `os.getenv(..., "default")` calls with `os.environ[...]` for security-critical parameters.
2.  **Environment Sanitization**: Create a unified `.env.example` and ensure `.env` is properly ignored.
3.  **Frontend Config Injection**: Move the frontend API key to `process.env.NEXT_PUBLIC_API_KEY` to allow build-time or runtime configuration.
4.  **CORS Bounding**: Remove all default CORS origins; require explicit environment configuration.
5.  **Error Scrubbing**: Sanitize the global exception handler to return a generic `Internal Server Error` message without class/library leakage.
