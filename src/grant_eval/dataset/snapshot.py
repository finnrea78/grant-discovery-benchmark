"""Freeze a funder's public page tree to a dated, hash-manifested snapshot.

Grid runs read from these snapshots, never the live web, so the bytes are kept as
served (no link rewriting) and hashed in ``manifest.json`` to catch any drift.

Usage::

    uv run snapshot --funder leverhulme \
        --entry https://www.leverhulme.ac.uk/ --out data/snapshots --depth 3
"""
from __future__ import annotations

import argparse
import hashlib
import html as _html
import json
import re
import shutil
import subprocess
import sys
from datetime import date as Date, datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

# Identifies the crawler and its purpose in funder server logs (crawler etiquette;
# no personal contact details are published to third-party sites).
USER_AGENT = (
    "msc-grant-eval/0.1 (Exeter MSc COMM424DA research crawler; "
    "frozen-snapshot evaluation; throttled; obeys robots.txt)"
)

# Content the agent can actually read. Assets (CSS/JS/images) are captured so pages
# render for human inspection, but they are not agent input and are not hashed.
PAGE_SUFFIXES = {".html", ".htm"}
DOC_SUFFIXES = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".txt", ".xml"}

_LOG_LINE = re.compile(r'URL:(\S+)\s+\[[^\]]*\]\s+->\s+"([^"]+)"')
_TITLE = re.compile(rb"<title[^>]*>(.*?)</title>", re.I | re.S)



def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, timeout: int = 30) -> bytes | None:
    """Single polite GET, used for the small out-of-band files (robots, sitemap)."""
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def normalise_url(url: str) -> str:
    """Scheme- and www-insensitive key for comparing crawled vs sitemap URLs."""
    p = urlparse(url)
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/") or "/"
    return f"{host}{path}"


def extract_title(path: Path) -> str | None:
    try:
        m = _TITLE.search(path.read_bytes()[:200_000])
    except OSError:
        return None
    if not m:
        return None
    raw = m.group(1).decode("utf-8", "replace")
    return _html.unescape(re.sub(r"\s+", " ", raw)).strip() or None



def build_wget_command(
    entry: str,
    site_dir: Path,
    log_path: Path,
    depth: int,
    wait: float,
    rate: str,
    no_parent: bool,
    reject_regex: str | None,
    assets: str,
) -> list[str]:
    host = urlparse(entry).netloc
    cmd = [
        "wget",
        "--recursive",
        f"--level={depth}",
        f"--domains={host}",
        "--adjust-extension",     # /schemes -> schemes.html on disk; bytes unchanged
        "--no-convert-links",     # deliberate: frozen input must stay byte-identical
        f"--user-agent={USER_AGENT}",
        f"--wait={wait}",
        "--random-wait",
        f"--limit-rate={rate}",
        "--tries=3",
        "--timeout=30",
        "--waitretry=5",
        "--no-verbose",
        f"--append-output={log_path}",
        f"--directory-prefix={site_dir}",
    ]
    # Assets are not agent input: on one funder they were 97% of the bytes (two
    # podcast MP3s alone outweighed every page). Default to skipping them.
    if assets != "none":
        cmd.append("--page-requisites")
    if assets == "images":
        cmd.append("--reject=mp3,mp4,m4a,wav,avi,mov,zip")
    if no_parent:
        cmd.append("--no-parent")
    if reject_regex:
        cmd.append(f"--reject-regex={reject_regex}")
    cmd.append(entry)
    return cmd


def parse_wget_log(log_path: Path) -> dict[str, str]:
    """Map local file path -> source URL, straight from wget's own log.

    More reliable than reconstructing URLs from the directory layout, because
    --adjust-extension and percent-decoding both alter the on-disk name.
    """
    mapping: dict[str, str] = {}
    if not log_path.exists():
        return mapping
    for line in log_path.read_text("utf-8", "replace").splitlines():
        m = _LOG_LINE.search(line)
        if m:
            mapping[str(Path(m.group(2)).resolve())] = m.group(1)
    return mapping


def derive_url(rel_to_site: Path) -> str:
    """Fallback URL for a file wget's log did not attribute (e.g. resumed runs)."""
    parts = rel_to_site.parts
    host, rest = parts[0], "/".join(parts[1:])
    return f"https://{host}/{rest}"



def build_manifest(snapshot_dir: Path, site_dir: Path, log_path: Path) -> dict:
    by_path = parse_wget_log(log_path)
    files: list[dict] = []

    for path in sorted(p for p in site_dir.rglob("*") if p.is_file()):
        rel_site = path.relative_to(site_dir)
        suffix = path.suffix.lower()
        if suffix in PAGE_SUFFIXES:
            kind = "page"
        elif suffix in DOC_SUFFIXES:
            kind = "document"
        else:
            kind = "asset"

        entry = {
            "path": str(path.relative_to(snapshot_dir)),
            "url": by_path.get(str(path.resolve())) or derive_url(rel_site),
            "kind": kind,
            "bytes": path.stat().st_size,
        }
        # Assets are captured for rendering only; they are not agent input, so they
        # are listed but not hashed or folded into the content hash.
        if kind != "asset":
            entry["sha256"] = sha256_file(path)
            if kind == "page":
                entry["title"] = extract_title(path)
        files.append(entry)

    hashed = sorted((f["path"], f["sha256"]) for f in files if "sha256" in f)
    content_hash = hashlib.sha256(
        "\n".join(f"{p} {h}" for p, h in hashed).encode()
    ).hexdigest()

    return {"content_hash": content_hash, "files": files}


def sitemap_coverage(entry: str, snapshot_dir: Path, captured: set[str]) -> dict:
    """How much of the funder's declared sitemap the crawl actually reached.

    A gap is either a deliberate exclusion (record it) or a hole in the crawl
    (fix it).
    """
    raw = fetch(urljoin(entry, "/sitemap.xml"))
    if not raw:
        return {"available": False}
    (snapshot_dir / "meta" / "sitemap.xml").write_bytes(raw)

    urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", raw.decode("utf-8", "replace"))
    missing = [u for u in urls if normalise_url(u) not in captured]
    return {
        "available": True,
        "declared": len(urls),
        "captured": len(urls) - len(missing),
        "missing_examples": missing[:25],
    }



def write_index(snapshot_dir: Path, meta: dict, manifest: dict) -> None:
    pages = [f for f in manifest["files"] if f["kind"] == "page"]
    docs = [f for f in manifest["files"] if f["kind"] == "document"]
    assets = [f for f in manifest["files"] if f["kind"] == "asset"]
    cov = meta.get("sitemap_coverage", {})

    lines = [
        f"# Snapshot index — {meta['funder']}",
        "",
        f"- **Snapshot date:** {meta['snapshot_date']}",
        f"- **Entry URL:** {meta['entry_url']}",
        f"- **Captured:** {meta['captured_at_utc']}",
        f"- **Content hash:** `{manifest['content_hash']}`",
        f"- **Pages:** {len(pages)} · **Documents:** {len(docs)} · "
        f"**Assets:** {len(assets)} · **Total:** {meta['total_bytes'] / 1e6:.1f} MB",
    ]
    if cov.get("available"):
        lines.append(
            f"- **Sitemap coverage:** {cov['captured']}/{cov['declared']} declared URLs"
        )
    lines += [
        "",
        "Original bytes as served; links are *not* rewritten, so open pages from the",
        "local paths below rather than by clicking through them.",
        "",
        f"## Pages ({len(pages)})",
        "",
        "| Title | URL | Local file |",
        "|---|---|---|",
    ]
    for f in sorted(pages, key=lambda f: f["url"]):
        title = (f.get("title") or "—").replace("|", "\\|")
        lines.append(f"| {title} | {f['url']} | `{f['path']}` |")

    lines += ["", f"## Documents ({len(docs)})", ""]
    if docs:
        lines += ["| URL | Local file | Size |", "|---|---|---|"]
        for f in sorted(docs, key=lambda f: f["url"]):
            lines.append(f"| {f['url']} | `{f['path']}` | {f['bytes'] / 1e3:.0f} kB |")
    else:
        lines.append("_None captured._")

    if cov.get("available") and cov.get("missing_examples"):
        lines += [
            "",
            f"## Declared but not captured ({cov['declared'] - cov['captured']})",
            "",
            "Sitemap URLs the crawl did not reach at this depth — triage these before",
            "labelling: each is either an accepted exclusion or a missed page.",
            "",
        ]
        lines += [f"- {u}" for u in cov["missing_examples"]]

    (snapshot_dir / "INDEX.md").write_text("\n".join(lines) + "\n")



def snapshot(args: argparse.Namespace) -> int:
    if not shutil.which("wget"):
        print("error: wget not found on PATH", file=sys.stderr)
        return 1

    snapshot_date = args.date or Date.today().isoformat()
    snapshot_dir = Path(args.out).resolve() / args.funder / snapshot_date
    site_dir, meta_dir = snapshot_dir / "site", snapshot_dir / "meta"

    if snapshot_dir.exists() and not args.overwrite:
        print(f"error: {snapshot_dir} exists (pass --overwrite to replace)", file=sys.stderr)
        return 1
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    site_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    # Due-diligence evidence, fetched at the same moment as the pages themselves.
    for name, url in (
        ("robots.txt", urljoin(args.entry, "/robots.txt")),
        ("terms.html", args.terms_url),
    ):
        if not url:
            continue
        body = fetch(url)
        if body:
            (meta_dir / name).write_bytes(body)
        else:
            print(f"warning: could not fetch {url}", file=sys.stderr)

    log_path = meta_dir / "wget.log"
    cmd = build_wget_command(
        args.entry, site_dir, log_path, args.depth, args.wait,
        args.rate, args.no_parent, args.reject_regex, args.assets,
    )
    print("$ " + " ".join(cmd), flush=True)
    started = datetime.now(timezone.utc)
    proc = subprocess.run(cmd)
    finished = datetime.now(timezone.utc)
    # wget exits 8 when any single URL 404s, which is normal on a real site.
    if proc.returncode not in (0, 8):
        print(f"error: wget exited {proc.returncode}", file=sys.stderr)
        return proc.returncode

    manifest = build_manifest(snapshot_dir, site_dir, log_path)
    captured = {normalise_url(f["url"]) for f in manifest["files"]}

    meta = {
        "funder": args.funder,
        "snapshot_date": snapshot_date,
        "entry_url": args.entry,
        "captured_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": finished.isoformat(timespec="seconds"),
        "duration_s": round((finished - started).total_seconds()),
        "crawl": {
            "tool": subprocess.run(
                ["wget", "--version"], capture_output=True, text=True
            ).stdout.splitlines()[0],
            "command": cmd,
            "depth": args.depth,
            "wait_s": args.wait,
            "rate_limit": args.rate,
            "assets_policy": args.assets,
            "robots_txt": "respected (wget default)",
            "link_rewriting": "disabled (frozen bytes)",
        },
        "counts": {
            k: sum(1 for f in manifest["files"] if f["kind"] == k)
            for k in ("page", "document", "asset")
        },
        "total_bytes": sum(f["bytes"] for f in manifest["files"]),
        "wget_exit_code": proc.returncode,
    }
    meta["sitemap_coverage"] = sitemap_coverage(args.entry, snapshot_dir, captured)

    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (meta_dir / "snapshot.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_index(snapshot_dir, meta, manifest)

    c = meta["counts"]
    print(
        f"\nfrozen: {snapshot_dir}\n"
        f"  {c['page']} pages, {c['document']} documents, {c['asset']} assets, "
        f"{meta['total_bytes'] / 1e6:.1f} MB in {meta['duration_s']}s\n"
        f"  content_hash {manifest['content_hash']}"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--funder", required=True, help="slug, e.g. leverhulme")
    p.add_argument("--entry", required=True, help="entry URL the crawl starts from")
    p.add_argument("--out", default="data/snapshots", help="snapshot root directory")
    p.add_argument("--date", help="snapshot_date (default: today, ISO)")
    p.add_argument("--depth", type=int, default=3, help="recursion depth from entry")
    p.add_argument("--wait", type=float, default=1.0, help="seconds between requests")
    p.add_argument("--rate", default="500k", help="wget --limit-rate value")
    p.add_argument("--no-parent", action="store_true", help="stay under the entry path")
    p.add_argument("--reject-regex", help="wget --reject-regex passthrough")
    p.add_argument("--assets", choices=("none", "images", "all"), default="none",
                   help="none: HTML + documents only (default); images: also page "
                        "requisites minus media; all: everything wget can reach")
    p.add_argument("--terms-url", help="terms/conditions page to capture as evidence")
    p.add_argument("--overwrite", action="store_true", help="replace an existing snapshot")
    return snapshot(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
