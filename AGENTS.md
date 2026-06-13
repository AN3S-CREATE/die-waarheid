# AGENTS.md

## Cursor Cloud specific instructions

Project "Die Waarheid": a forensic WhatsApp/audio analysis platform. Three dev services:

| Service | Port | Dev command (from repo root) | Notes |
| --- | --- | --- | --- |
| FastAPI backend | 8000 | `cd die_waarheid && ../venv/bin/python -m uvicorn api_server:app --host 0.0.0.0 --port 8000` | Must run from `die_waarheid/` (see caveat) |
| React frontend | 3000 | `cd frontend && npm run dev` | Vite dev server is on **3000** (per `vite.config.ts`), not 5173 |
| Streamlit UI | 8501 | `./venv/bin/python -m streamlit run die_waarheid/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true` | Primary in-process UI |

Python deps live in a repo-local virtualenv at `./venv` (gitignored, persisted in the VM snapshot). Always invoke Python via `./venv/bin/python`.

### Non-obvious caveats

- **Run the backend from `die_waarheid/`.** `api_server.py` uses top-level `from src...` / `from config...` imports, so `uvicorn die_waarheid.api_server:app` from the repo root fails. Run `uvicorn api_server:app` with cwd = `die_waarheid/` (the unified `die_waarheid/launcher.py` is unreliable for this reason).
- **`setuptools` must be `<81`.** setuptools 81+ removed `pkg_resources`, which `librosa`/`audioread` import at audio-load time. The update script pins `setuptools<81` last.
- **`.env` files are required and not committed** (gitignored). Three copies are used: repo-root `.env`, `die_waarheid/.env` (same content), and `frontend/.env`. `frontend/.env` `VITE_API_KEY` MUST equal the backend `API_KEY`, and `VITE_API_URL` must point to `http://localhost:8000`. If `.env` is missing, recreate from `.env.example` and generate an `API_KEY` (`python -c "import secrets;print(secrets.token_urlsafe(32))"`).
- **CPU-only by default.** No GPU in this environment; keep `USE_GPU=false` and `WHISPER_MODEL_SIZE=tiny` in `.env` for fast transcription. The tiny Whisper model downloads on first `/api/transcribe` call.
- **pyannote diarization is disabled** (incompatible with the installed `huggingface-hub`); the code logs "Pyannote not available" and falls back to energy-based diarization. This is expected and non-fatal.
- **Advanced security file scanner false-positives on binary audio.** With `bleach` installed, `src/security.py::_scan_file_content` flags any upload whose raw bytes contain sequences like `<%`; `/api/transcribe` and `/api/analyze` may return `"File contains potentially malicious content"` for otherwise-valid audio. Re-encode/regenerate the file if testing through the API.

### Known pre-existing issues (NOT environment problems — do not "fix" as setup)

- `ForensicsEngine.load_audio` raises `unhashable type: 'numpy.ndarray'` (`weakref.WeakSet().add(ndarray)`), so the `/api/analyze` forensic path is broken in the repo.
- `pytest tests/` has ~24 failing tests due to test/code drift (mocked `librosa`, sanitizer behavior, etc.); ~22 pass. Coverage gate (`--cov-fail-under=80` in `pytest.ini`) will also fail — run with `--no-cov` to see raw pass/fail.
- Frontend production build (`npm run build` → `tsc -b`) fails on pre-existing strict TS errors (unused imports, type-only import). The dev server (`npm run dev`) does not typecheck and runs fine.

### Lint / test commands

- Backend lint (CI critical subset): `./venv/bin/flake8 die_waarheid --select=E9,F63,F7,F82` (CI also uses `black`/`isort`/`mypy`; install on demand).
- Frontend lint: `cd frontend && npm run lint`.
- Tests: `./venv/bin/python -m pytest tests/ --no-cov -q`.

### Verified hello-world

Speech WAV → React Transcribe page (English, Tiny) → FastAPI `/api/transcribe` → Whisper returned `"forensic analysis system online."` end-to-end on CPU.
