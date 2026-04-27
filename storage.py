"""
Versioned transcript storage on the filesystem.

Layout per recording:
  data/transcripts/{id}/index.json            { source, active, versions[] }
  data/transcripts/{id}/{version}.txt         transcript text
  data/transcripts/{id}/{version}.summary.txt summary (optional)

Legacy flat-file layout is migrated on first read.
"""

import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import config


def _rec_dir(rec_id: str) -> Path:
    return Path(config.TRANSCRIPTS_DIR) / rec_id


def _index_path(rec_id: str) -> Path:
    return _rec_dir(rec_id) / "index.json"


def _read_index(rec_id: str) -> dict:
    p = _index_path(rec_id)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_index(rec_id: str, index: dict) -> None:
    _rec_dir(rec_id).mkdir(parents=True, exist_ok=True)
    _index_path(rec_id).write_text(json.dumps(index), encoding="utf-8")


def _legacy_paths(rec_id: str) -> tuple[Path, Path, Path]:
    return (
        Path(config.TRANSCRIPTS_DIR) / f"{rec_id}.txt",
        Path(config.TRANSCRIPTS_DIR) / f"{rec_id}.json",
        Path(config.SUMMARIES_DIR) / f"{rec_id}.txt",
    )


def _active_version(index: dict) -> Optional[dict]:
    active_id = index.get("active")
    for v in index.get("versions", []):
        if v.get("id") == active_id:
            return v
    return None


def _new_version_id() -> str:
    return f"v{int(time.time() * 1000)}"


def migrate_legacy(rec_id: str) -> None:
    """Convert flat-file recording into the versioned directory layout."""
    legacy_txt, legacy_meta, legacy_summary = _legacy_paths(rec_id)
    if _index_path(rec_id).exists():
        # Already versioned; clean up any leftover flat files
        legacy_txt.unlink(missing_ok=True)
        legacy_meta.unlink(missing_ok=True)
        return
    if not legacy_txt.exists():
        return

    meta = {}
    if legacy_meta.exists():
        try:
            meta = json.loads(legacy_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    rec_dir = _rec_dir(rec_id)
    rec_dir.mkdir(parents=True, exist_ok=True)
    version_id = _new_version_id()
    (rec_dir / f"{version_id}.txt").write_text(
        legacy_txt.read_text(encoding="utf-8"), encoding="utf-8"
    )

    version = {
        "id": version_id,
        "model": meta.get("model", ""),
        "created_at": meta.get("created_at", datetime.now().isoformat()),
        "speakers": meta.get("speakers", {}),
        "speakers_list": meta.get("speakers_list", []),
        "has_summary": False,
    }
    if legacy_summary.exists():
        (rec_dir / f"{version_id}.summary.txt").write_text(
            legacy_summary.read_text(encoding="utf-8"), encoding="utf-8"
        )
        version["has_summary"] = True

    index = {
        "source": meta.get("source", ""),
        "active": version_id,
        "versions": [version],
    }
    _write_index(rec_id, index)

    legacy_txt.unlink(missing_ok=True)
    legacy_meta.unlink(missing_ok=True)
    legacy_summary.unlink(missing_ok=True)


def list_recordings() -> list[dict]:
    transcripts_dir = Path(config.TRANSCRIPTS_DIR)
    if not transcripts_dir.exists():
        return []

    # Lazy-migrate any legacy stragglers
    for txt in transcripts_dir.glob("*.txt"):
        migrate_legacy(txt.stem)

    results = []
    for rec_dir in transcripts_dir.iterdir():
        if not rec_dir.is_dir():
            continue
        index = _read_index(rec_dir.name)
        active = _active_version(index)
        if not active:
            continue
        results.append({
            "id": rec_dir.name,
            "source": index.get("source", ""),
            "model": active.get("model", ""),
            "created_at": active.get("created_at", ""),
            "has_summary": bool(active.get("has_summary")),
            "version_count": len(index.get("versions", [])),
        })
    results.sort(key=lambda x: x["created_at"], reverse=True)
    return results


def get_recording(rec_id: str) -> Optional[dict]:
    migrate_legacy(rec_id)
    index = _read_index(rec_id)
    active = _active_version(index)
    if not active:
        return None
    rec_dir = _rec_dir(rec_id)
    text_path = rec_dir / f"{active['id']}.txt"
    if not text_path.exists():
        return None
    summary = None
    summary_path = rec_dir / f"{active['id']}.summary.txt"
    if summary_path.exists():
        summary = summary_path.read_text(encoding="utf-8")
    return {
        "id": rec_id,
        "source": index.get("source", ""),
        "active": active["id"],
        "text": text_path.read_text(encoding="utf-8"),
        "speakers": active.get("speakers", {}),
        "speakers_list": active.get("speakers_list", []),
        "model": active.get("model", ""),
        "created_at": active.get("created_at", ""),
        "summary": summary,
        "versions": [
            {
                "id": v["id"],
                "model": v.get("model", ""),
                "created_at": v.get("created_at", ""),
                "has_summary": bool(v.get("has_summary")),
            }
            for v in index.get("versions", [])
        ],
    }


def add_version(rec_id: str, text: str, model_label: str, source: str) -> str:
    """Append a new version, set it active, return its id."""
    rec_dir = _rec_dir(rec_id)
    rec_dir.mkdir(parents=True, exist_ok=True)
    migrate_legacy(rec_id)
    index = _read_index(rec_id)

    version_id = _new_version_id()
    (rec_dir / f"{version_id}.txt").write_text(text, encoding="utf-8")
    speakers_list = sorted(set(re.findall(r"SPEAKER_\d+", text)))

    version = {
        "id": version_id,
        "model": model_label,
        "created_at": datetime.now().isoformat(),
        "speakers": {},
        "speakers_list": speakers_list,
        "has_summary": False,
    }

    if not index:
        index = {"source": source, "active": version_id, "versions": [version]}
    else:
        index.setdefault("versions", []).append(version)
        index["active"] = version_id
        if source and not index.get("source"):
            index["source"] = source

    _write_index(rec_id, index)
    return version_id


def set_active(rec_id: str, version_id: str) -> bool:
    index = _read_index(rec_id)
    if not index:
        return False
    if not any(v.get("id") == version_id for v in index.get("versions", [])):
        return False
    index["active"] = version_id
    _write_index(rec_id, index)
    return True


def update_speakers(rec_id: str, speakers: dict) -> bool:
    index = _read_index(rec_id)
    active = _active_version(index)
    if not active:
        return False
    active["speakers"] = speakers
    _write_index(rec_id, index)
    return True


def update_summary(rec_id: str, summary: str) -> bool:
    index = _read_index(rec_id)
    active = _active_version(index)
    if not active:
        return False
    rec_dir = _rec_dir(rec_id)
    (rec_dir / f"{active['id']}.summary.txt").write_text(summary, encoding="utf-8")
    active["has_summary"] = True
    _write_index(rec_id, index)
    return True


def delete_recording(rec_id: str) -> None:
    rec_dir = _rec_dir(rec_id)
    if rec_dir.exists():
        shutil.rmtree(rec_dir)
    for p in _legacy_paths(rec_id):
        p.unlink(missing_ok=True)
