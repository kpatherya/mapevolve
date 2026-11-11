"""Download the demo-world images listed in data/demo_world/sources.yaml.

    python -m scripts.fetch_wikimedia                 # 1600px images -> data/demo_world/images/
    python -m scripts.fetch_wikimedia --width 320 --out /tmp/thumbs   # quick preview

Only files whose Commons license is CC0, public domain, CC BY or CC BY-SA are
accepted (no NC/ND). For each file the source page, author, license, license
URL, capture date and SHA-256 are written to data/demo_world/attribution.json.
Requests use a generic client name and back off on HTTP 429.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo_world"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "MapEvolveDemoFetcher/0.1 (research prototype; python-urllib)"}


def license_ok(name: str) -> bool:
    n = name.upper()
    if "NC" in n.replace("NC-", "NC ").split() or "-NC" in n or "-ND" in n or " ND" in n:
        return False
    return n.startswith(("CC0", "CC BY", "PUBLIC DOMAIN", "PD"))


def _get(url: str, binary: bool = False, attempts: int = 6) -> Any:
    for i in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                data = r.read()
                return data if binary else json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or i == attempts - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 15 * (i + 1))
            print(f"  rate limited, waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def image_info(titles: list[str], width: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(titles), 40):
        params = {
            "action": "query", "format": "json", "formatversion": 2, "maxlag": 5,
            "titles": "|".join(titles[i : i + 40]), "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1|extmetadata", "iiurlwidth": width,
            "iiextmetadatafilter": "Artist|LicenseShortName|LicenseUrl|Credit|DateTimeOriginal|AttributionRequired|GPSLatitude|GPSLongitude",
        }
        data = _get(f"{API}?{urllib.parse.urlencode(params)}")
        norm = {n["to"]: n["from"] for n in data["query"].get("normalized", [])}
        for page in data["query"]["pages"]:
            out[norm.get(page["title"], page["title"])] = page
        time.sleep(1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, default=DEMO / "sources.yaml")
    parser.add_argument("--out", type=Path, default=DEMO / "images")
    parser.add_argument("--width", type=int, default=1600)
    args = parser.parse_args()

    specs = yaml.safe_load(args.sources.read_text(encoding="utf-8"))["images"]
    info = image_info([s["title"] for s in specs], args.width)
    args.out.mkdir(parents=True, exist_ok=True)
    records, problems = [], []
    for spec in specs:
        page = info.get(spec["title"])
        if not page or "imageinfo" not in page:
            problems.append(f"{spec['id']}: not found on Commons ({spec['title']})")
            continue
        ii = page["imageinfo"][0]
        meta = {k: _clean(v.get("value", "")) for k, v in ii.get("extmetadata", {}).items()}
        lic = meta.get("LicenseShortName", "")
        if not license_ok(lic):
            problems.append(f"{spec['id']}: license {lic!r} not allowed")
            continue
        target = args.out / f"{spec['id']}.jpg"
        if not target.exists():
            target.write_bytes(_get(ii.get("thumburl") or ii["url"], binary=True))
            time.sleep(1.0)
        data = target.read_bytes()
        records.append({
            "id": spec["id"],
            "path": str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target),
            "commons_title": spec["title"],
            "source_page": ii.get("descriptionurl"),
            "author": meta.get("Artist") or meta.get("Credit"),
            "license": lic,
            "license_url": meta.get("LicenseUrl"),
            "attribution_required": meta.get("AttributionRequired"),
            "date_original": meta.get("DateTimeOriginal"),
            "gps": [meta.get("GPSLatitude"), meta.get("GPSLongitude")] if meta.get("GPSLatitude") else None,
            "original_size": [ii.get("width"), ii.get("height")],
            "downloaded_width": args.width,
            "sha256": hashlib.sha256(data).hexdigest(),
        })
        print(f"ok {spec['id']:<28} {lic:<14} {len(data) // 1024} KB", flush=True)
    if args.out == DEMO / "images":
        (DEMO / "attribution.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in problems:
        print("SKIP", p, file=sys.stderr)
    total = sum((args.out / f"{r['id']}.jpg").stat().st_size for r in records)
    print(f"{len(records)} images, {total / 1e6:.1f} MB -> {args.out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
