"""Download Poker44 benchmark releases into this miner's data/benchmark/ folder.

Self-contained: writes release_<date>.json into <repo>/data/benchmark/ (gitignored)
so the miner can retrain from a fresh clone without any external folder. By
default it fetches only releases not already on disk -- run it daily to pull the
new day. Override the target with --output-dir or POKER44_BENCHMARK_DIR.

    python detector/fetch_benchmark.py            # fetch any new releases
    python detector/fetch_benchmark.py --force    # re-download everything
    python detector/fetch_benchmark.py --dates 2026-07-17
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://api.poker44.net/api/v1/benchmark"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(os.environ.get("POKER44_BENCHMARK_DIR", str(ROOT / "data" / "benchmark")))


def _get_json(url: str, timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not payload.get("success", True):
        raise RuntimeError(f"API error for {url}: {payload}")
    return payload.get("data", payload)


def list_all_release_dates(*, page_size: int = 100) -> list[str]:
    dates: list[str] = []
    before: str | None = None
    while True:
        params: dict[str, str | int] = {"limit": page_size}
        if before:
            params["before"] = before
        page = _get_json(f"{API_BASE}/releases?{urllib.parse.urlencode(params)}")
        releases = page.get("releases") or []
        if not releases:
            break
        for item in releases:
            if isinstance(item, dict) and item.get("sourceDate"):
                dates.append(str(item["sourceDate"]))
        before = str(releases[-1].get("sourceDate") or "")
        if len(releases) < page_size:
            break
    return sorted(set(dates))


def fetch_release(source_date: str, *, limit: int = 24) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    cursor: str | None = None
    release_meta: dict[str, Any] = {}
    while True:
        params: dict[str, str | int] = {"sourceDate": source_date, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        page = _get_json(f"{API_BASE}/chunks?{urllib.parse.urlencode(params)}")
        if not release_meta:
            release_meta = {k: page.get(k) for k in
                            ("sourceDate", "releaseVersion", "schemaVersion", "releaseType", "split")
                            if k in page}
        page_chunks = page.get("chunks") or []
        if isinstance(page_chunks, list):
            chunks.extend(item for item in page_chunks if isinstance(item, dict))
        cursor = page.get("nextCursor")
        if not cursor:
            break
    return {**release_meta, "chunks": chunks}


def _count_labeled_examples(payload: dict[str, Any]) -> tuple[int, int]:
    examples = hands = 0
    for group in payload.get("chunks") or []:
        if not isinstance(group, dict):
            continue
        inner = group.get("chunks") or []
        labels = group.get("groundTruth") or group.get("groundTruthLabels") or []
        examples += min(len(inner), len(labels))
        for chunk in inner:
            if isinstance(chunk, list):
                hands += len(chunk)
    return examples, hands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Poker44 benchmark releases (incremental).")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dates", type=str, default=None,
                        help="Comma-separated YYYY-MM-DD dates (default: all available).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if release_<date>.json already exists.")
    parser.add_argument("--chunk-limit", type=int, default=24, help="Chunks per API page (max 24).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    status = _get_json(API_BASE)
    print("API corpus:", f"totalChunks={status.get('totalChunks')}",
          f"latestSourceDate={status.get('latestSourceDate')}")

    dates = [d.strip() for d in (args.dates or "").split(",") if d.strip()] or list_all_release_dates()
    if not dates:
        raise RuntimeError("No benchmark release dates found.")

    new, skipped = 0, 0
    for source_date in dates:
        target = output_dir / f"release_{source_date}.json"
        if target.exists() and not args.force:
            skipped += 1
            continue
        try:
            payload = fetch_release(source_date, limit=args.chunk_limit)
            examples, hands = _count_labeled_examples(payload)
            if examples <= 0:
                print(f"WARN: skipped {source_date}: no labeled examples")
                continue
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            new += 1
            print(f"saved {target.name}: examples={examples} hands={hands}")
        except urllib.error.HTTPError as err:
            print(f"WARN: skipped {source_date}: HTTP {err.code}")
        except Exception as err:  # pragma: no cover
            print(f"WARN: skipped {source_date}: {err}")

    print(f"Done: {new} new release(s), {skipped} already present -> {output_dir}")


if __name__ == "__main__":
    main()
