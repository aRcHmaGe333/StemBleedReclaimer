# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import BleedCleaner


def main() -> int:
    parser = argparse.ArgumentParser(description="Attribute and conservatively remove cross-stem bleed.")
    parser.add_argument("stem_folder", type=Path, help="Folder containing drums.wav, bass.wav, other.wav, vocals.wav")
    parser.add_argument("output_folder", type=Path, help="New folder for cleaned stems and evidence")
    args = parser.parse_args()

    result = BleedCleaner().process_folder(args.stem_folder, args.output_folder)
    report = args.output_folder / "bleed_attribution_report.json"
    report.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"regions={len(result.regions)}")
    print(f"report={report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

