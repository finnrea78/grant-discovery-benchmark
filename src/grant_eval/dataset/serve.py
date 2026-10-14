"""Serve a frozen snapshot over HTTP, resolving live URLs to mirrored files.

The mirror keeps the original bytes, so root-absolute links only resolve through the
source URLs in ``manifest.json``. The agent's fetch tool points here at eval time.
Content is served byte-exact; ``serve_all`` is the better tool for browsing and
labelling. Requests not in the mirror are logged as MISS.

Usage::

    uv run serve --snapshot data/snapshots/leverhulme/2026-08-09
    uv run serve_all                                 # every snapshot, for browsing
"""
from __future__ import annotations

import argparse
import json
import mimetypes
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlparse

_LOG_LINE = re.compile(r'URL:(\S+)\s+\[[^\]]*\]\s+->\s+"([^"]+)"')


def build_lookup(snapshot_dir: Path, manifest: dict) -> dict[tuple[str, str], Path]:
    """Index every captured file under the URL forms a client might ask for."""
    lookup: dict[tuple[str, str], Path] = {}
    for entry in manifest["files"]:
        parsed = urlparse(entry["url"])
        path = parsed.path or "/"
        target = snapshot_dir / entry["path"]
        # Most specific first; setdefault keeps it from being shadowed by a
        # less-specific key from another file.
        for key in (
            (path, parsed.query),
            (path, ""),
            (path.rstrip("/") or "/", ""),
        ):
            lookup.setdefault(key, target)
    return lookup


def resolve(lookup: dict[tuple[str, str], Path], request_path: str, query: str) -> Path | None:
    # Try both spellings of the path. wget names the file after the decoded path
    # ("divi_layout_injector 3.5/style.css") while the manifest, and so the lookup
    # key, keeps the encoded URL, which is also what a browser requests.
    raw = request_path or "/"
    decoded = unquote(raw)
    # Both directions: a request may arrive already decoded (a rewritten link that
    # went through html.unescape) while the key is encoded, or vice versa.
    forms = list(dict.fromkeys([decoded, raw, quote(decoded, safe="/~")]))

    for path in forms:
        for key in (
            (path, query),
            (path, ""),
            (path.rstrip("/") or "/", ""),
            (path + "/", ""),
            (path + ".html", ""),
            (path.rstrip("/") + "/index.html", ""),
        ):
            hit = lookup.get(key)
            if hit and hit.is_file():
                return hit

    # The site serves some pages at more than one casing (/Early-Career-Fellowships
    # and /early-career-fellowships); match case-insensitively as a last resort.
    lowered = {f.lower() for f in forms}
    for (kp, kq), target in lookup.items():
        if kq == "" and kp.lower() in lowered and target.is_file():
            return target
    return None


def make_handler(lookup, scope_prefix: str | None, misses: Counter):
    class SnapshotHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            if scope_prefix and not parsed.path.startswith(scope_prefix):
                self._fail(403, f"outside served scope {scope_prefix!r}")
                return

            target = resolve(lookup, parsed.path, parsed.query)
            if target is None:
                misses[parsed.path] += 1
                self._fail(404, "not in this snapshot")
                return

            body = target.read_bytes()
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype == "application/xhtml+xml":
                ctype += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def _fail(self, code: int, msg: str) -> None:
            body = f"{code} — {msg}\n".encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            print(f"  {fmt % args}", flush=True)

    return SnapshotHandler


def manifest_from_wget_log(snapshot_dir: Path) -> dict | None:
    """Reconstruct a manifest-shaped index from wget's log.

    Only the fields the server needs (path, url, kind); no hashes, since a partial
    crawl has no meaningful content hash and must never be mistaken for a frozen one.
    """
    log_path = snapshot_dir / "meta" / "wget.log"
    if not log_path.exists():
        return None

    files = []
    for line in log_path.read_text("utf-8", "replace").splitlines():
        m = _LOG_LINE.search(line)
        if not m:
            continue
        target = Path(m.group(2))
        if not target.is_file():
            continue
        suffix = target.suffix.lower()
        files.append(
            {
                "path": str(target.resolve().relative_to(snapshot_dir)),
                "url": m.group(1),
                "kind": "page" if suffix in (".html", ".htm") else "asset",
            }
        )
    return {"content_hash": "PARTIAL-CRAWL-NOT-FROZEN", "files": files}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshot", required=True, help="path to <funder>/<snapshot_date>/")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--scope-prefix", help="only serve URLs under this path prefix")
    args = p.parse_args()

    snapshot_dir = Path(args.snapshot).resolve()
    manifest_path = snapshot_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        # The manifest is only written when a crawl completes. Fall back to wget's
        # log so a partial or interrupted snapshot can still be browsed.
        manifest = manifest_from_wget_log(snapshot_dir)
        if manifest is None:
            print(f"error: no manifest.json or meta/wget.log in {snapshot_dir}")
            return 1
        print("note: no manifest.json yet — serving from meta/wget.log (crawl in progress?)")
    lookup = build_lookup(snapshot_dir, manifest)
    misses: Counter = Counter()

    pages = sum(1 for f in manifest["files"] if f["kind"] == "page")
    print(
        f"serving {snapshot_dir.parent.name} @ {snapshot_dir.name}\n"
        f"  {len(manifest['files'])} files ({pages} pages), "
        f"content_hash {manifest['content_hash'][:12]}…\n"
        f"  http://localhost:{args.port}/    (ctrl-c to stop)"
    )
    if args.scope_prefix:
        print(f"  served scope restricted to {args.scope_prefix}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(lookup, args.scope_prefix, misses))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if misses:
            print(f"\n{len(misses)} URL(s) requested but not in the snapshot:")
            for url, n in misses.most_common(20):
                print(f"  {n:>3}x {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
