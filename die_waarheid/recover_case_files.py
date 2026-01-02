import json
import os
import re
import shutil
from pathlib import Path


def _safe_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "case"
    s = re.sub(r"[^A-Za-z0-9._ -]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120] if len(s) > 120 else s


def _unique_path(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    session_path = repo_root / "session_data.json"
    if not session_path.exists():
        print(f"ERROR: session_data.json not found at: {session_path}")
        return 1

    data = json.loads(session_path.read_text(encoding="utf-8"))

    case_id = data.get("case_id") or "case"
    safe_case = _safe_name(case_id)

    case_root = repo_root / "data" / "cases" / safe_case
    evidence_dir = case_root / "evidence"
    speakers_dir = case_root / "speakers"

    evidence_dir.mkdir(parents=True, exist_ok=True)
    speakers_dir.mkdir(parents=True, exist_ok=True)

    evidence_files = data.get("evidence_files") or []

    total = len(evidence_files)
    exists = 0
    copied = 0
    missing = 0

    for i, f in enumerate(evidence_files, start=1):
        src = f.get("path")
        if not src or not os.path.exists(src):
            missing += 1
            continue

        exists += 1
        src_path = Path(src)
        dest_path = _unique_path(evidence_dir / src_path.name)

        if dest_path.exists():
            f["path"] = str(dest_path)
            continue

        shutil.copy2(src_path, dest_path)
        f["path"] = str(dest_path)
        copied += 1

        if i % 250 == 0:
            print(f"Recovered {i}/{total}...")

    speaker_profiles = data.get("speaker_profiles") or {}
    speaker_copied = 0
    for name, profile in speaker_profiles.items():
        sample_file = profile.get("sample_file")
        if not sample_file or not os.path.exists(sample_file):
            continue
        src_path = Path(sample_file)
        dest_path = _unique_path(speakers_dir / src_path.name)
        shutil.copy2(src_path, dest_path)
        profile["sample_file"] = str(dest_path)
        speaker_copied += 1

    session_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DONE")
    print(f"Case: {case_id} -> {case_root}")
    print(f"Evidence total: {total}")
    print(f"Evidence exists: {exists}")
    print(f"Evidence copied: {copied}")
    print(f"Evidence missing: {missing}")
    print(f"Speaker samples copied: {speaker_copied}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
