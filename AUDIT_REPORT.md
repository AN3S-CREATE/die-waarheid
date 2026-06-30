# Die Waarheid — Senior-Level Repository Audit Report

**Date:** 2026-05-13  
**Auditor:** Blackbox AI (Senior Code Review)  
**Scope:** Full codebase — Python backend, FastAPI server, Streamlit app, tests, Docker, config

---

## 1. Executive Summary

Die Waarheid is a forensic WhatsApp communication analysis platform built on Python 3.11. It combines Whisper speech-to-text, librosa audio forensics, pyannote speaker diarization, and Gemini/HuggingFace AI for psychological profiling. The architecture is sound in concept, but the codebase has **several confirmed bugs**, significant **security gaps**, **test/implementation mismatches**, and **dependency health concerns** that must be addressed before production deployment.

**Overall Risk Rating: HIGH** — Multiple confirmed bugs, unauthenticated endpoints, and test suite that tests a different API than what is implemented.

---

## 2. Critical Fixes (Confirmed Bugs)

### BUG-01 — Duplicate Route Registration (FastAPI Crash / Silent Override)
**File:** `die_waarheid/api_server.py`, lines ~230 and ~310  
**Severity:** CRITICAL  
**Evidence:**
```python
@app.get("/api/security/status")
@limiter.limit("10/minute")
async def security_status(request: Request, api_key: str = Depends(verify_api_key)):
    ...

# ... 80 lines later ...

@app.get("/api/security/status")   # ← EXACT DUPLICATE
@limiter.limit("10/minute")
async def security_status(request: Request, api_key: str = Depends(verify_api_key)):
    ...
```
FastAPI silently uses the **first** registered route and ignores the second. This is a copy-paste error. The second definition is dead code and will cause confusion.

**Fix:**
```python
# Remove the second duplicate definition of security_status entirely.
# Keep only one definition (the first occurrence, ~line 230).
```

---

### BUG-02 — `chat_parser.py` — Loop Variable Scope Bug (Messages Lost)
**File:** `die_waarheid/src/chat_parser.py`, `parse_file_async()`, lines ~80–100  
**Severity:** CRITICAL  
**Evidence:**
```python
for i in range(0, len(lines), batch_size):
    batch = lines[i:i + batch_size]
    
    for line in batch:
        if not line.strip():
            continue
        parsed = self._parse_line(line)

    # BUG: `parsed` and `current_message` logic is OUTSIDE the inner for-loop
    if parsed and parsed.get('is_new_message'):   # ← indented at batch level, not line level
        if current_message:
            self.messages.append(current_message)
        current_message = parsed
        ...
    elif current_message and not parsed.get('is_new_message'):
        current_message['text'] += '\n' + line    # ← `line` is last line of batch, not current
```
The `if parsed` block is indented at the **batch** level, not the **line** level. Only the **last line of each batch** is ever processed for message assembly. All other lines are silently discarded. This means the parser loses the vast majority of messages.

**Fix:**
```python
for line in batch:
    if not line.strip():
        continue
    parsed = self._parse_line(line)
    
    if parsed and parsed.get('is_new_message'):   # ← move inside inner loop
        if current_message:
            self.messages.append(current_message)
        current_message = parsed
        if parsed.get('sender'):
            self.participants.add(parsed['sender'])
    elif current_message and not parsed.get('is_new_message'):
        current_message['text'] += '\n' + line
```

---

### BUG-03 — `database.py` — `get_session()` Method Name Collision
**File:** `die_waarheid/src/database.py`, `DatabaseManager` class  
**Severity:** CRITICAL  
**Evidence:**
```python
@contextmanager
def get_session(self):          # ← context manager version (line ~280)
    session = self.SessionLocal()
    try:
        yield session
        session.commit()
    ...

def get_session(self) -> Session:   # ← plain getter version (line ~300) — OVERWRITES the above
    """Get database session"""
    return self.SessionLocal()
```
Python class definitions are sequential — the second `get_session` **completely replaces** the first. All callers that use `with self.get_session() as session:` will get a `Session` object (not a context manager) and will receive an `AttributeError: __enter__`. Meanwhile, `store_analysis_result`, `store_message`, `store_conversation_analysis`, and `store_psychological_profile` all call `self.get_session()` without a context manager and **never close or commit** the session, causing connection leaks.

**Fix:**
```python
# Rename the plain getter to avoid collision:
def get_raw_session(self) -> Session:
    """Get raw database session (caller is responsible for close/commit)"""
    return self.SessionLocal()

# Keep the context manager as get_session() and use session_scope() for the alias.
# Update store_* methods to use session_scope():
def store_analysis_result(self, case_id: str, result: dict) -> bool:
    try:
        with self.session_scope() as session:
            analysis = AnalysisResult(...)
            session.add(analysis)
        return True
    except Exception as e:
        logger.error(f"Error storing analysis result: {str(e)}")
        return False
```

---

### BUG-04 — `database.py` — Invalid SQLAlchemy Aggregation Call
**File:** `die_waarheid/src/database.py`, `get_case_statistics()`, line ~390  
**Severity:** HIGH  
**Evidence:**
```python
avg_stress = session.query(AnalysisResult).filter(
    AnalysisResult.case_id == case_id
).with_entities(
    AnalysisResult.stress_level.avg()   # ← AttributeError: Column has no .avg()
).scalar()
```
SQLAlchemy columns do not have an `.avg()` method. This will raise `AttributeError` at runtime.

**Fix:**
```python
from sqlalchemy import func

avg_stress = session.query(
    func.avg(AnalysisResult.stress_level)
).filter(
    AnalysisResult.case_id == case_id
).scalar()
```

---

### BUG-05 — `api_server.py` — `/api/analyze` Endpoint Has No Authentication
**File:** `die_waarheid/api_server.py`, line ~340  
**Severity:** HIGH  
**Evidence:**
```python
@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):   # ← No api_key dependency
    """Perform forensic audio analysis"""
```
The `/api/transcribe` endpoint requires `api_key: str = Depends(verify_api_key)`, but `/api/analyze`, `/api/speakers`, `/api/speakers/initialize`, `/api/speakers/train`, and `/api/files/count` have **no authentication at all**. Any unauthenticated user can upload files and trigger expensive ML inference.

**Fix:**
```python
@app.post("/api/analyze")
@limiter.limit("10/minute")
async def analyze_audio(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)   # ← add auth
):
```
Apply the same pattern to all unprotected endpoints.

---

### BUG-06 — `api_server.py` — Temp File Not Cleaned Up on Analysis Error
**File:** `die_waarheid/api_server.py`, `analyze_audio()`, lines ~345–365  
**Severity:** HIGH  
**Evidence:**
```python
async def analyze_audio(file: UploadFile = File(...)):
    ...
    with tempfile.NamedTemporaryFile(...) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    
    result = forensics_engine.analyze(tmp_path)
    tmp_path.unlink()   # ← Only called on success; if analyze() raises, file leaks
```
Unlike `/api/transcribe` which uses `try/finally`, `/api/analyze` only deletes the temp file on the happy path. Any exception leaves the file on disk.

**Fix:**
```python
tmp_path = None
try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    result = forensics_engine.analyze(tmp_path)
    return result
except HTTPException:
    raise
except Exception as e:
    logger.error(f"Analysis error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
finally:
    if tmp_path and tmp_path.exists():
        tmp_path.unlink()
```

---

### BUG-07 — `whisper_transcriber.py` — Model Loaded But Never Cached
**File:** `die_waarheid/src/whisper_transcriber.py`, `load_model()`, lines ~90–130  
**Severity:** HIGH  
**Evidence:**
```python
def load_model(self) -> bool:
    cache_key = f"{self.model_size}_{self.device}"
    
    with self._cache_lock:
        if cache_key in self._model_cache:
            self.model = self._model_cache[cache_key]
            return True
    
    # ... loads model ...
    self.model = whisper.load_model(self.model_size, device=device)
    # BUG: self._model_cache[cache_key] = self.model is NEVER called
    return True
```
The class-level `_model_cache` dict is checked on load but **never written to**. Every new `WhisperTranscriber` instance reloads the model from disk, defeating the caching mechanism entirely.

**Fix:**
```python
self.model = whisper.load_model(self.model_size, device=device)

# Cache the loaded model
with self._cache_lock:
    self._model_cache[cache_key] = self.model

logger.info(f"Successfully loaded and cached Whisper {self.model_size} model")
return True
```

---

### BUG-08 — `cache.py` — MD5 Used for Security-Sensitive Cache Keys
**File:** `die_waarheid/src/cache.py`, `get_file_hash()`, line ~50  
**Severity:** MEDIUM  
**Evidence:**
```python
def get_file_hash(self, file_path: Path) -> str:
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
```
MD5 is cryptographically broken. While this is used as a cache key (not for security), it creates collision risk for large audio files. Additionally, the entire file is read into memory to compute the hash — for 500MB files this is a significant memory spike.

**Fix:**
```python
def get_file_hash(self, file_path: Path) -> str:
    """Generate SHA-256 hash using chunked reading to avoid memory spikes."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()
```

---

### BUG-09 — `security.py` — Injection Pattern Blocks Legitimate Afrikaans Text
**File:** `die_waarheid/src/security.py`, `INJECTION_PATTERNS`, line ~35  
**Severity:** MEDIUM  
**Evidence:**
```python
r"(?i)(;|\||&|`|\$\(|\${|<\(|>\()",   # Command injection patterns
```
The semicolon `;` is flagged as a command injection character. Afrikaans text commonly uses semicolons in sentences. Any message containing a semicolon will be rejected with HTTP 400 "Potentially malicious input detected", breaking the core use case.

**Fix:**
```python
# Remove the overly broad semicolon match; keep only shell-specific patterns:
r"(?i)(\||&|`|\$\(|\${|<\(|>\()",   # Remove `;` from this pattern
```

---

### BUG-10 — `speaker_identification.py` — `_extract_pitch_features()` Unpacks Wrong Return Value
**File:** `die_waarheid/src/speaker_identification.py` / `die_waarheid/src/forensics.py`  
**Severity:** MEDIUM  
**Evidence:**
In `forensics.py`:
```python
def _extract_pitch_features(self) -> Tuple[float, float]:
    f0, voiced_flag = self.extract_pitch()   # extract_pitch returns (f0, times) — 2 values
    voiced_f0 = f0[voiced_flag]              # voiced_flag is actually `times` array — wrong!
```
`extract_pitch()` returns `(f0, times)` but `_extract_pitch_features()` unpacks it as `(f0, voiced_flag)`. The boolean mask indexing `f0[voiced_flag]` will either fail or produce garbage results.

**Fix:**
```python
def _extract_pitch_features(self) -> Tuple[float, float]:
    f0, times = self.extract_pitch()
    valid_f0 = f0[~np.isnan(f0)]   # Use NaN mask instead of voiced_flag
    if len(valid_f0) == 0:
        return 0.0, 0.0
    return float(np.mean(valid_f0)), float(np.std(valid_f0))
```

---

## 3. Medium-Priority Improvements

### IMP-01 — `api_server.py` — Leaked API Key in Logs
**File:** `die_waarheid/api_server.py`, lines ~55–57  
**Severity:** MEDIUM  
**Evidence:**
```python
if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    logger.warning(f"No API_KEY found in environment. Generated temporary key: {API_KEY}")
```
The generated API key is logged in plaintext. Any log aggregation system (CloudWatch, Datadog, etc.) will capture and store this secret.

**Fix:**
```python
logger.warning("No API_KEY found in environment. Generated a temporary key. Set API_KEY env var for production!")
# Do NOT log the key value
```

---

### IMP-02 — `ai_analyzer.py` — LRU Cache Keyed on Mutable Text (Cache Poisoning Risk)
**File:** `die_waarheid/src/ai_analyzer.py`, `_init_cache()` and `analyze_message()`  
**Severity:** MEDIUM  
**Evidence:**
```python
self._cached_analyze = lru_cache(maxsize=self.cache_size)(self._analyze_uncached)
# Called as:
result = self._cached_analyze(text_hash, text)
```
`lru_cache` is applied to an instance method, which means `self` is part of the cache key. This prevents the cache from working across instances. Additionally, the `text` argument (potentially thousands of characters) is included in the cache key alongside `text_hash`, making the hash redundant and wasting memory.

**Fix:**
```python
# Use a dict-based cache keyed only on text_hash:
self._response_cache: Dict[str, Dict] = {}

def analyze_message(self, text: str) -> Dict:
    text = self.sanitize_input(text)
    text_hash = self._get_text_hash(text)
    
    if text_hash in self._response_cache:
        self.cache_hits += 1
        return {**self._response_cache[text_hash], 'cached': True}
    
    self.cache_misses += 1
    result = self._analyze_uncached(text_hash, text)
    self._response_cache[text_hash] = result
    return result
```

---

### IMP-03 — `pipeline_processor.py` — `ForensicsEngine` Not Thread-Safe in Parallel Batch
**File:** `die_waarheid/src/pipeline_processor.py`, `process_batch()` and `process_voice_note()`  
**Severity:** MEDIUM  
**Evidence:**
```python
self.forensics = ForensicsEngine(use_cache=False)  # Single shared instance

def process_batch(self, audio_files, ...):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(self.process_voice_note, audio_path, ...): audio_path
            ...
        }
```
`ForensicsEngine` stores `self.audio_data` and `self.filename` as instance state. Multiple threads calling `self.forensics.analyze()` concurrently will overwrite each other's audio data, producing corrupted results.

**Fix:**
```python
def process_voice_note(self, audio_path, language, model_size):
    # Create a per-call forensics engine instead of sharing one:
    forensics = ForensicsEngine(use_cache=False)
    forensic_result = forensics.analyze(audio_path)
    ...
```

---

### IMP-04 — `config.py` — Logging Configured at Module Import Time
**File:** `die_waarheid/config.py`, lines ~160–170  
**Severity:** MEDIUM  
**Evidence:**
```python
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
```
`logging.basicConfig()` is called at module import time. Any module that imports `config` before the application sets up its own logging will have its configuration overridden. This also creates a file handler unconditionally, even in test environments.

**Fix:**
```python
def configure_logging():
    """Call this explicitly from app entry points, not at import time."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
```

---

### IMP-05 — `security.py` — In-Memory Rate Limit Storage Grows Unbounded
**File:** `die_waarheid/src/security.py`, `_rate_limit_storage` dict  
**Severity:** MEDIUM  
**Evidence:**
```python
_rate_limit_storage: Dict[str, List[float]] = {}

def check_rate_limit(self, client_ip, endpoint, ...):
    key = f"{client_ip}:{endpoint}"
    if key not in _rate_limit_storage:
        _rate_limit_storage[key] = []
    # Old entries are cleaned per-key, but the dict itself grows forever
    _rate_limit_storage[key] = [req_time for req_time in _rate_limit_storage[key] if ...]
```
Every unique `(IP, endpoint)` combination creates a permanent dict entry. Under load with many unique IPs, this dict grows without bound, causing a memory leak.

**Fix:**
```python
# Periodically evict stale keys:
def _cleanup_rate_limit_storage():
    current_time = time.time()
    stale_keys = [k for k, v in _rate_limit_storage.items() 
                  if not v or current_time - max(v) > RATE_LIMIT_WINDOW * 2]
    for k in stale_keys:
        del _rate_limit_storage[k]
```

---

### IMP-06 — `speaker_identification.py` — SQLite Engine Created Without Thread Safety
**File:** `die_waarheid/src/speaker_identification.py`, `__init__()`, line ~200  
**Severity:** MEDIUM  
**Evidence:**
```python
self.engine = create_engine(f'sqlite:///{db_path}')
# Missing: connect_args={"check_same_thread": False}
```
SQLite requires `check_same_thread=False` when used from multiple threads. The `SpeakerIdentificationSystem` is used from the FastAPI async context and potentially from thread pools, which will cause `ProgrammingError: SQLite objects created in a thread can only be used in that same thread`.

**Fix:**
```python
self.engine = create_engine(
    f'sqlite:///{db_path}',
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
```

---

### IMP-07 — `chat_parser.py` — `asyncio.run()` Called Inside Potentially Running Event Loop
**File:** `die_waarheid/src/chat_parser.py`, `parse_file()`, line ~55  
**Severity:** MEDIUM  
**Evidence:**
```python
def parse_file(self, file_path: Path) -> Tuple[bool, str]:
    return asyncio.run(self.parse_file_async(file_path))
```
`asyncio.run()` creates a new event loop. If called from within an already-running async context (e.g., from a FastAPI endpoint or Streamlit), this raises `RuntimeError: This event loop is already running`.

**Fix:**
```python
def parse_file(self, file_path: Path) -> Tuple[bool, str]:
    try:
        loop = asyncio.get_running_loop()
        # Already in async context — run synchronously
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, self.parse_file_async(file_path))
            return future.result()
    except RuntimeError:
        return asyncio.run(self.parse_file_async(file_path))
```
Or better: make `parse_file` a fully synchronous implementation and keep `parse_file_async` for async callers.

---

### IMP-08 — `forensics.py` — `batch_analyze_parallel()` Shares Mutable State
**File:** `die_waarheid/src/forensics.py`, `batch_analyze_parallel()`  
**Severity:** MEDIUM  
**Evidence:**
```python
def batch_analyze_parallel(self, file_paths, max_workers=4, ...):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(self.analyze, fp): fp for fp in file_paths}
```
`self.analyze()` modifies `self.audio_data` and `self.filename`. Calling it from multiple threads on the same `ForensicsEngine` instance causes race conditions.

**Fix:** Same as IMP-03 — create per-call engine instances, or add a threading lock around audio data access.

---

### IMP-09 — `api_server.py` — `asyncio.get_event_loop()` Deprecated
**File:** `die_waarheid/api_server.py`, `transcribe_audio()`, line ~295  
**Severity:** LOW-MEDIUM  
**Evidence:**
```python
loop = asyncio.get_event_loop()
result = await asyncio.wait_for(
    loop.run_in_executor(executor, transcriber.transcribe, tmp_path, language),
    timeout=300
)
```
`asyncio.get_event_loop()` is deprecated in Python 3.10+ and raises a `DeprecationWarning`. In Python 3.12 it may raise `RuntimeError` if no current event loop exists.

**Fix:**
```python
result = await asyncio.wait_for(
    asyncio.get_event_loop().run_in_executor(executor, transcriber.transcribe, tmp_path, language),
    timeout=300
)
# Or better:
import asyncio
result = await asyncio.wait_for(
    asyncio.to_thread(transcriber.transcribe, tmp_path, language),
    timeout=300
)
```

---

## 4. Upgrade Opportunities

### UPG-01 — `google-generativeai==0.3.2` is Severely Outdated
**File:** `requirements.txt`  
**Severity:** HIGH  
**Evidence:** The current version is `0.3.2` (released ~2023). The current stable release is `0.8.x`/`1.x`. The `gemini-2.0-flash` model referenced in `config.py` (`GEMINI_MODEL = "gemini-2.0-flash"`) was released in 2025 and requires the newer SDK. The old SDK will raise `AttributeError` or `ValueError` when trying to use this model.

**Fix:**
```
google-generativeai>=0.8.0
```

---

### UPG-02 — `pydantic==2.5.0` with `@validator` (Pydantic v1 API)
**File:** `die_waarheid/api_server.py`, `requirements.txt`  
**Severity:** MEDIUM  
**Evidence:**
```python
from pydantic import BaseModel, Field, validator   # validator is deprecated in Pydantic v2

class TranscriptionRequest(BaseModel):
    @validator('language')
    def validate_language(cls, v):
        ...
```
`@validator` is a Pydantic v1 API. In Pydantic v2 it still works via compatibility shim but emits `DeprecationWarning`. The correct v2 API is `@field_validator`.

**Fix:**
```python
from pydantic import BaseModel, Field, field_validator

class TranscriptionRequest(BaseModel):
    @field_validator('language')
    @classmethod
    def validate_language(cls, v):
        if v not in ['af', 'en', 'nl']:
            raise ValueError('Language must be af, en, or nl')
        return v
```

---

### UPG-03 — `openai-whisper==20250625` — Future-Dated Version
**File:** `requirements.txt`  
**Severity:** MEDIUM  
**Evidence:** The pinned version `openai-whisper==20250625` has a date of June 25, 2025. As of the audit date (May 2026), this may or may not be the latest. More importantly, pinning to a date-versioned package without a hash makes supply-chain verification impossible.

**Recommendation:** Use `pip-compile` with `--generate-hashes` to lock all dependencies with integrity hashes.

---

### UPG-04 — `torch==2.5.1` / `torchaudio==2.5.1` — Outdated
**File:** `requirements.txt`  
**Severity:** LOW-MEDIUM  
**Evidence:** PyTorch 2.6.x is available with significant performance improvements for inference. The current pin may miss security patches.

**Recommendation:** Test with `torch>=2.6.0` and update if compatible.

---

### UPG-05 — `@app.on_event("startup")` Deprecated in FastAPI
**File:** `die_waarheid/api_server.py`, line ~155  
**Severity:** LOW  
**Evidence:**
```python
@app.on_event("startup")
async def startup_event():
```
`on_event` is deprecated since FastAPI 0.93. The modern approach uses `lifespan` context managers.

**Fix:**
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    await startup_event()
    yield
    # shutdown (add cleanup here)

app = FastAPI(lifespan=lifespan, ...)
```

---

### UPG-06 — No `requirements-lock.txt` / `pip-compile` Workflow
**File:** `requirements.txt`  
**Severity:** MEDIUM  
**Evidence:** Dependencies are pinned at the top level but transitive dependencies are not locked. A `pip install -r requirements.txt` on a fresh machine may install different transitive versions, causing subtle bugs.

**Recommendation:**
```bash
pip install pip-tools
pip-compile requirements.txt --output-file requirements.lock
# Use requirements.lock in Docker and CI
```

---

## 5. Testing Recommendations

### TEST-01 — Unit Tests Test a Different API Than What Exists
**File:** `tests/unit/test_forensics.py`  
**Severity:** HIGH  
**Evidence:**
```python
# test_forensics.py calls:
engine._calculate_stress_level(mean_pitch=200.0, pitch_std=50.0, silence_ratio=0.4)
engine._calculate_pitch_volatility(50.0)
engine._assess_audio_quality(duration=60.0, signal_to_noise=30.0, zero_crossing_rate=0.1)
engine.get_summary_statistics()
engine._extract_intensity_features()  # expects dict with 'mean', 'std', 'max'

# But forensics.py actually has:
engine._calculate_stress_level(pitch_volatility, silence_ratio, intensity_max, mfcc_variance)
# No _calculate_pitch_volatility(), _assess_audio_quality(), get_summary_statistics() methods
# _extract_intensity_features() returns a float, not a dict
```
The entire test suite for `ForensicsEngine` tests methods that **do not exist** in the implementation. These tests will all fail with `AttributeError`. The tests appear to have been written against a different version of the code.

**Fix:** Rewrite `tests/unit/test_forensics.py` to match the actual `ForensicsEngine` API.

---

### TEST-02 — No Integration Tests for API Endpoints
**File:** `tests/integration/` (empty or missing)  
**Severity:** HIGH  
**Evidence:** The `tests/integration/` directory exists but contains no test files. The FastAPI endpoints (`/api/transcribe`, `/api/analyze`, `/api/speakers/*`) have zero test coverage.

**Recommendation:**
```python
# tests/integration/test_api.py
from fastapi.testclient import TestClient
from die_waarheid.api_server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_transcribe_requires_auth():
    response = client.post("/api/transcribe", files={"file": b"fake"})
    assert response.status_code == 403  # or 401

def test_analyze_requires_auth():
    response = client.post("/api/analyze", files={"file": b"fake"})
    assert response.status_code == 403  # Currently fails — no auth on this endpoint
```

---

### TEST-03 — `conftest.py` Mocks `librosa` Globally (Breaks Real Tests)
**File:** `tests/conftest.py`, `disable_api_calls` fixture  
**Severity:** MEDIUM  
**Evidence:**
```python
@pytest.fixture(autouse=True)
def disable_api_calls(monkeypatch):
    mock_librosa = Mock()
    mock_librosa.load = Mock(return_value=(Mock(), 22050))
    sys.modules['librosa'] = mock_librosa
```
This `autouse=True` fixture replaces `librosa` with a `Mock()` for **every test**, including tests that need real librosa behavior. Tests like `test_load_audio_success` patch `src.forensics.librosa` separately, creating a conflict.

**Fix:** Remove the global librosa mock from `conftest.py`. Let individual tests mock what they need.

---

### TEST-04 — No Tests for `chat_parser.py` (Critical Bug Undetected)
**File:** `tests/`  
**Severity:** HIGH  
**Evidence:** There are no tests for `WhatsAppParser`. The critical indentation bug (BUG-02) went undetected because there is no test that verifies multi-line message parsing or batch processing.

**Recommendation:**
```python
# tests/unit/test_chat_parser.py
def test_parse_multiline_message(tmp_path):
    chat_file = tmp_path / "chat.txt"
    chat_file.write_text(
        "01/01/2024, 10:00 - Alice: Hello\nthis is line 2\n"
        "01/01/2024, 10:01 - Bob: Hi there\n"
    )
    parser = WhatsAppParser()
    success, msg = parser.parse_file(chat_file)
    assert success
    assert len(parser.get_messages()) == 2
    assert "line 2" in parser.get_messages()[0]['text']
```

---

### TEST-05 — Missing Tests for Security Module
**File:** `tests/`  
**Severity:** MEDIUM  
**Evidence:** `src/security.py` has no unit tests. The injection detection patterns, file magic number validation, and rate limiting logic are untested.

---

## 6. Security Findings Summary

| Finding | Location | Severity |
|---------|----------|----------|
| API key logged in plaintext | `api_server.py:57` | HIGH |
| 5 endpoints with no authentication | `api_server.py` | HIGH |
| Semicolon blocks Afrikaans text | `security.py:35` | MEDIUM |
| MD5 used for file hashing | `cache.py:50` | MEDIUM |
| Rate limit storage memory leak | `security.py` | MEDIUM |
| Gemini safety settings all BLOCK_NONE | `config.py` | LOW |
| No HTTPS enforcement in dev mode | `api_server.py` | LOW |
| `bleach` not in `requirements.txt` | `security.py` imports bleach | HIGH |

**Critical:** `security.py` imports `bleach` (`import bleach`) but `bleach` is not listed in `requirements.txt`. This will cause an `ImportError` at startup, disabling the entire advanced security module silently (caught by the `try/except ImportError` in `api_server.py`).

**Fix:** Add to `requirements.txt`:
```
bleach==6.1.0
```

---

## 7. Suggested Implementation Roadmap

### Phase 1 — Critical Bugs (Fix Immediately, ~1–2 days)
1. **BUG-02** — Fix chat parser indentation bug (data loss)
2. **BUG-03** — Fix `get_session()` name collision in `DatabaseManager`
3. **BUG-04** — Fix invalid `.avg()` SQLAlchemy call
4. **BUG-01** — Remove duplicate `/api/security/status` route
5. **BUG-07** — Fix Whisper model cache write
6. Add `bleach` to `requirements.txt`

### Phase 2 — Security Hardening (~2–3 days)
7. **BUG-05** — Add authentication to all unprotected endpoints
8. **BUG-06** — Fix temp file cleanup in `/api/analyze`
9. **IMP-01** — Remove API key from log output
10. **BUG-09** — Fix semicolon injection false positive
11. **IMP-05** — Add rate limit storage cleanup

### Phase 3 — Correctness & Stability (~3–5 days)
12. **BUG-10** — Fix `_extract_pitch_features()` unpacking
13. **IMP-03 / IMP-08** — Fix thread safety in parallel processing
14. **IMP-06** — Fix SQLite thread safety in speaker identification
15. **IMP-07** — Fix `asyncio.run()` in sync `parse_file()`
16. **BUG-08** — Replace MD5 with SHA-256 chunked hashing

### Phase 4 — Test Suite Repair (~2–3 days)
17. **TEST-01** — Rewrite `test_forensics.py` to match actual API
18. **TEST-02** — Add integration tests for all API endpoints
19. **TEST-03** — Fix global librosa mock in conftest
20. **TEST-04** — Add `WhatsAppParser` unit tests
21. **TEST-05** — Add security module tests

### Phase 5 — Dependency & Tooling Upgrades (~1–2 days)
22. **UPG-01** — Upgrade `google-generativeai` to `>=0.8.0`
23. **UPG-02** — Migrate `@validator` to `@field_validator` (Pydantic v2)
24. **UPG-05** — Migrate `on_event` to `lifespan` (FastAPI)
25. **UPG-06** — Add `pip-compile` lockfile workflow
26. **IMP-09** — Replace deprecated `asyncio.get_event_loop()`

---

## 8. Quick Wins (< 30 minutes each)

| Item | File | Change |
|------|------|--------|
| Add `bleach` to requirements | `requirements.txt` | 1 line |
| Remove duplicate route | `api_server.py` | Delete ~20 lines |
| Fix API key log leak | `api_server.py:57` | Remove `{API_KEY}` from f-string |
| Fix SQLAlchemy avg() call | `database.py:390` | Use `func.avg()` |
| Add auth to `/api/analyze` | `api_server.py` | Add `Depends(verify_api_key)` |
| Fix Whisper cache write | `whisper_transcriber.py` | Add 2 lines |
| Fix temp file cleanup | `api_server.py` | Wrap in try/finally |

---

*End of Audit Report*
