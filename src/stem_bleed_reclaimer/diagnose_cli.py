# PO-Signed: OpenAI Codex 2026-07-15
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .leakage_diagnostics import LeakageRelationDiagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure whether quiet-stem bleed is scaled, delayed, filtered, or independently transferred.")
    parser.add_argument("stem_folders", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnostic = LeakageRelationDiagnostic()
    report = {}
    for folder in args.stem_folders:
        report[folder.name] = {lane: result.to_dict() for lane, result in diagnostic.analyze_folder(folder).items()}
    report["__pooled_leave_one_track_out__"] = {
        lane: result.to_dict() for lane, result in diagnostic.analyze_pooled(args.stem_folders).items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
