"""Audit a staged public release for common publication blockers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", "outputs", ".pytest_cache", "__pycache__"}
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yml", ".yaml"}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "generic_secret": re.compile(r"(?i)(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*[^\s$<{]{8,}"),
    "absolute_windows_path": re.compile(r"(?i)\b[A-Z]:\\(?:Users|Documents and Settings)\\"),
}
# Construct these strings in pieces so the audit does not flag its own rules.
RELEASE_PLACEHOLDERS = ("REPLACE-" + "WITH-ACCOUNT", "ReefFormer " + "study team")


def included_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.as_posix() == "MANIFEST.sha256":
            continue
        files.append(path)
    return sorted(files, key=lambda value: value.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--allow-release-placeholders", action="store_true")
    args = parser.parse_args()

    findings = []
    placeholders = []
    large_files = []
    files = included_files()
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.stat().st_size >= 50 * 1024 * 1024:
            large_files.append({"path": relative, "bytes": path.stat().st_size})
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({"path": relative, "kind": label, "offset": match.start()})
        for placeholder in RELEASE_PLACEHOLDERS:
            if placeholder in text:
                placeholders.append({"path": relative, "value": placeholder})

    if args.write_manifest:
        lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
        (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")

    report = {
        "files_audited": len(files),
        "secret_or_path_findings": findings,
        "release_placeholders": placeholders,
        "files_at_least_50_mb": large_files,
        "manifest_written": bool(args.write_manifest),
    }
    report["status"] = (
        "PASS"
        if not findings and not large_files and (args.allow_release_placeholders or not placeholders)
        else "BLOCKED"
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
