"""Serve every funder snapshot at once, one origin per funder, plus a hub index.

Mirrored pages use root-absolute links, so each site needs its own origin: one port
per funder. The hub (base port) lists every snapshot and shows each ground-truth CSV
with links to the local copy of each labelled page. Links are rewritten to stay on
localhost; ``--strict`` also stops uncaptured assets loading from the live site.

Usage::

    uv run serve_all                              # hub on :8100, funders :8101+
    uv run serve_all --strict                     # never touch the live web
    uv run serve_all --port 9000 --data data/snapshots
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import re
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from grant_eval.dataset.serve import build_lookup, manifest_from_wget_log, resolve

CSS = """
:root{--bg:#fff;--fg:#16181d;--muted:#5c6370;--line:#e3e6ea;--accent:#1a56b8;--warn:#8a5a00;--warnbg:#fff7e6;--ok:#0a6b3d;--okbg:#eaf7f0;--card:#fafbfc}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaed;--muted:#9aa2ad;--line:#2b2f36;--accent:#7fa8f0;--warn:#e0b062;--warnbg:#2c2413;--ok:#7bd3a4;--okbg:#13291f;--card:#1a1d22}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.5rem 4rem;background:var(--bg);color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:1600px}
h1{font-size:1.5rem;margin:0 0 .25rem} h2{font-size:1.1rem;margin:2rem 0 .5rem}
a{color:var(--accent)} .sub{color:var(--muted);margin:0 0 1.5rem;font-size:.9rem}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.03em;
  position:sticky;top:0;background:var(--bg);white-space:nowrap}
tbody tr:hover{background:var(--card)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
.pill{display:inline-block;padding:.1rem .45rem;border-radius:99px;font-size:.75rem;white-space:nowrap}
.pill.ok{background:var(--okbg);color:var(--ok)} .pill.pend{background:var(--warnbg);color:var(--warn)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.8rem}
.nowrap{white-space:nowrap} .num{text-align:right;font-variant-numeric:tabular-nums}
.note{color:var(--muted);font-size:.85rem;max-width:70ch}
.back{display:inline-block;margin-bottom:1rem;font-size:.85rem}
td.wrap{min-width:22ch;max-width:46ch}
"""


def origins_for(manifest: dict) -> list[str]:
    """Every host this snapshot was captured from, plus its www/non-www twin.

    Longest first, so ``www.example.org`` is replaced before a bare ``example.org``
    could match the tail of it.
    """
    hosts: set[str] = set()
    for entry in manifest["files"]:
        netloc = urlparse(entry["url"]).netloc
        if not netloc:
            continue
        hosts.add(netloc)
        hosts.add(netloc[4:] if netloc.startswith("www.") else "www." + netloc)
    return sorted(hosts, key=len, reverse=True)


_A_TAG = re.compile(rb"<a\b[^>]*>", re.I)


def _apexes(hosts: list[str]) -> list[str]:
    """The registrable domains this snapshot actually captured.

    Derived from the captured host list rather than guessed with a public-suffix
    rule, so ``www.wolfson.org.uk`` yields ``wolfson.org.uk`` and never a bare
    ``org.uk`` that would swallow every other charity's site.
    """
    stripped = {h[4:] if h.startswith("www.") else h for h in hosts}
    return sorted(
        (h for h in stripped if not any(o != h and h.endswith("." + o) for o in stripped)),
        key=len,
        reverse=True,
    )


def _host_re(hosts: list[str], escaped: bool) -> re.Pattern[bytes]:
    """Match an absolute URL pointing at one of this snapshot's own hosts.

    Sibling subdomains of a captured apex count as own-domain too. RSC links out to
    ``members.rsc.org`` and loads a tag manager from ``analytics.rsc.org``, so treating
    those as third-party sent clicks to the live site and fired tracking beacons. Two
    sibling hosts sharing a path collide onto one local page, which is better than
    leaving the mirror.

    ``escaped`` covers the ``https:\\/\\/host\\/path`` form that inline JSON
    (WordPress block data, JSON-LD) uses.
    """
    apexes = _apexes(hosts)
    parts = [rb"(?:[A-Za-z0-9-]+\.)*" + re.escape(a.encode()) for a in apexes]
    parts += [
        re.escape(h.encode())
        for h in hosts
        if not any(h == a or h.endswith("." + a) for a in apexes)
    ]
    alt = b"|".join(parts)
    if escaped:
        return re.compile(rb"https?:\\/\\/(?:" + alt + rb")((?:\\/|[^\"'\s>)\\])*)")
    return re.compile(rb"(?:https?:)?//(?:" + alt + rb")([^\"'\s>)]*)")


def _path_query(raw: bytes) -> tuple[str, str]:
    text = html.unescape(raw.decode("utf-8", "replace").replace("\\/", "/")) or "/"
    parsed = urlparse(text)
    return parsed.path or "/", parsed.query


def rewrite_html(
    body: bytes, site: dict, is_css: bool = False, strict: bool = False
) -> tuple[bytes, int]:
    """Point the snapshot's own absolute URLs at its local origin.

    Returns ``(body, n_left_live)``, where the count is own-domain URLs left pointing
    at the live web, shown in the page badge.

    Links (``<a href>``) are always made local, captured or not, because labelling
    from today's live content would corrupt the ground truth; an uncaptured link gets
    the "not in this snapshot" page. Assets (css, js, images) are made local only when
    captured, otherwise they load live so the page stays readable. The bytes on disk
    are untouched and the eval-time server (``serve``) does none of this.
    """
    local = f"http://localhost:{site['port']}".encode()
    plain, escaped = site["re_plain"], site["re_escaped"]
    captured = site["captured"]
    left_live = 0

    def always(m: re.Match[bytes]) -> bytes:
        return local + m.group(1)

    def only_if_captured(m: re.Match[bytes]) -> bytes:
        nonlocal left_live
        if captured(*_path_query(m.group(1))):
            return local + m.group(1)
        left_live += 1
        return m.group(0)

    if not is_css:
        # Pass 1: inside <a ...> tags, always rewritten.
        out = bytearray()
        pos = 0
        for m in _A_TAG.finditer(body):
            out += body[pos : m.start()] + plain.sub(always, m.group(0))
            pos = m.end()
        out += body[pos:]
        body = bytes(out)

    # Pass 2: everything else. Already-rewritten links no longer match these.
    # --strict drops the live fallback: nothing at all leaves localhost, at the cost
    # of pages whose stylesheets were never captured rendering bare or blank.
    rule = always if strict else only_if_captured
    body = escaped.sub(rule, plain.sub(rule, body))
    return body, left_live


def badge(funder: str, date: str, port: int, hub_port: int, live: int) -> bytes:
    """A corner marker: which snapshot this page came from, and what it borrowed live.

    ``live`` is the count of this page's own-domain URLs still pointing at the real
    site (uncaptured css/js/images). Text always comes from the snapshot; this says
    how much of what you are looking at did not.
    """
    s = (
        "position:fixed;bottom:12px;right:12px;z-index:2147483647;"
        "background:#16181d;color:#e8eaed;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;"
        "padding:6px 10px;border-radius:6px;opacity:.88;box-shadow:0 2px 8px rgba(0,0,0,.3);"
        "max-width:60vw"
    )
    live_note = (
        f' · <span style="color:#e0b062" title="uncaptured assets loaded from the live site; '
        f'page text is still from the snapshot">{live} live asset{"s" if live != 1 else ""}</span>'
        if live
        else ' · <span style="color:#7bd3a4">fully offline</span>'
    )
    return (
        f'<div style="{s}">snapshot · {html.escape(funder)} · {date} · :{port}{live_note} '
        f'<a href="http://localhost:{hub_port}/" target="_top" '
        f'style="color:#7fa8f0;text-decoration:underline">hub</a></div>'
    ).encode()


def guard(hosts: list[str]) -> bytes:
    """Keep runtime-generated links on localhost, which the byte rewrite cannot reach.

    ``rewrite_html`` only fixes URLs present in the served bytes. Some sites build
    links in the browser (RSC's funding search widget, for example), so this script
    intercepts clicks and rewrites own-domain targets in place. Browsing only.
    """
    apexes = json.dumps(_apexes(hosts))
    return (
        "<script>(function(){var A=" + apexes + ";"
        "function own(h){h=(h||'').toLowerCase();"
        "for(var i=0;i<A.length;i++){if(h===A[i]||h.slice(-(A[i].length+1))==='.'+A[i])return true}return false}"
        "function loc(u){try{var x=new URL(u,location.href);if(!own(x.hostname))return null;"
        "return location.origin+x.pathname+x.search+x.hash}catch(e){return null}}"
        "document.addEventListener('click',function(e){var n=e.target;"
        "while(n&&n.nodeType===1&&n.tagName!=='A'){n=n.parentNode}"
        "if(!n||n.nodeType!==1||!n.getAttribute)return;var h=n.getAttribute('href');if(!h)return;"
        "var u=loc(h);if(u){n.setAttribute('href',u);if(n.target==='_blank'){n.target='_self'}}},true);"
        "var O=window.open;window.open=function(u){var v=u?loc(u):null;"
        "return O.apply(window,[v||u].concat([].slice.call(arguments,1)))};})();</script>"
    ).encode()


def inject(body: bytes, marker: bytes) -> bytes:
    idx = body.lower().rfind(b"</body>")
    return body[:idx] + marker + body[idx:] if idx != -1 else body + marker


def page(title: str, body: str) -> bytes:
    return (
        f"<!doctype html><html><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    ).encode()


def load_manifest(snapshot_dir: Path) -> dict | None:
    mf = snapshot_dir / "manifest.json"
    if mf.exists():
        return json.loads(mf.read_text())
    return manifest_from_wget_log(snapshot_dir)


def discover(snapshots_root: Path, gt_root: Path) -> list[dict]:
    """One entry per funder: newest snapshot date, its manifest, its ground-truth CSV."""
    out = []
    for funder_dir in sorted(p for p in snapshots_root.iterdir() if p.is_dir()):
        dates = sorted(p for p in funder_dir.iterdir() if p.is_dir())
        if not dates:
            continue
        snap = dates[-1]
        manifest = load_manifest(snap)
        if manifest is None:
            continue
        meta = snap / "meta" / "snapshot.json"
        entry_url = ""
        if meta.exists():
            try:
                entry_url = json.loads(meta.read_text()).get("entry_url") or ""
            except json.JSONDecodeError:
                pass
        gt = next(iter(sorted(gt_root.glob(f"{funder_dir.name}_*.csv"))), None)
        rows = []
        if gt is not None:
            with gt.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        out.append(
            {
                "funder": funder_dir.name,
                "date": snap.name,
                "dir": snap,
                "manifest": manifest,
                "entry_url": entry_url,
                "gt_path": gt,
                "gt_rows": rows,
                "other_dates": [d.name for d in dates[:-1]],
            }
        )
    return out


def landing_path(site: dict, captured) -> str:
    """The page the hub should open this funder at, one the snapshot actually holds.

    Several funders were crawled from a subpath with ``--no-parent``, so ``/`` was never
    captured. Try the crawl's entry URL, then ``/``, then the shallowest captured page.
    """
    candidates: list[tuple[str, str]] = []
    entry = urlparse(site.get("entry_url") or "")
    if entry.path:
        candidates.append((entry.path, entry.query))
    candidates.append(("/", ""))
    pages = sorted(
        (urlparse(f["url"]) for f in site["manifest"]["files"] if f.get("kind") == "page"),
        key=lambda u: (u.path.count("/"), len(u.path)),
    )
    candidates.extend(((u.path or "/"), u.query) for u in pages)

    for path, query in candidates:
        if captured(path, query):
            return f"{path}?{query}" if query else path
    return "/"


def local_url(site: dict, url: str, source_page: str) -> str | None:
    """Map a ground-truth row back to its page on this funder's local origin."""
    base = f"http://localhost:{site['port']}"
    if url:
        parsed = urlparse(url)
        if parsed.path or parsed.query:
            q = f"?{parsed.query}" if parsed.query else ""
            return f"{base}{parsed.path or '/'}{q}"
        return base + "/"
    if source_page:
        # site/www.example.org/foo.html -> look the file's own URL up in the manifest
        for entry in site["manifest"]["files"]:
            if entry["path"] == source_page:
                return local_url(site, entry["url"], "")
    return None


def hub_index(sites: list[dict], base_port: int) -> bytes:
    total_pages = sum(s["n_pages"] for s in sites)
    labelled = sum(1 for s in sites if s["gt_rows"])
    rows = []
    for s in sites:
        gt_n = len(s["gt_rows"])
        pill = (
            f"<span class='pill ok'>{gt_n} row{'s' if gt_n != 1 else ''}</span>"
            if gt_n
            else "<span class='pill pend'>header only</span>"
        )
        gt_link = (
            f"<a href='/gt/{s['funder']}'>ground truth</a>" if gt_n else "<span class=note>—</span>"
        )
        host = s["host"] or "(unknown host)"
        extra = (
            f"<div class=note>older: {', '.join(s['other_dates'])}</div>" if s["other_dates"] else ""
        )
        landing = s.get("landing", "/")
        deep = (
            f"<div class=note>opens at <span class=mono>{html.escape(landing)}</span> "
            f"— site root not in this crawl</div>"
            if landing != "/"
            else ""
        )
        rows.append(
            f"<tr><td class=wrap><a href='http://localhost:{s['port']}{landing}' target=_blank>"
            f"<strong>{html.escape(s['funder'])}</strong></a>"
            f"<div class='note mono'>{html.escape(host)}</div>{deep}{extra}</td>"
            f"<td class=nowrap>{s['date']}</td>"
            f"<td class='num'>{s['n_pages']}</td><td class='num'>{s['n_docs']}</td>"
            f"<td class='num'>{s['n_files']}</td>"
            f"<td class=nowrap>{s['style_cell']}</td>"
            f"<td class=nowrap>{pill}</td>"
            f"<td class=nowrap><a href='http://localhost:{s['port']}{landing}' target=_blank>"
            f"open :{s['port']}</a> · <a href='/pages/{s['funder']}'>page list</a>"
            f" · {gt_link}</td></tr>"
        )
    body = (
        f"<h1>msc-code snapshots</h1>"
        f"<p class=sub>{len(sites)} funder snapshots · {total_pages} pages · "
        f"{labelled}/{len(sites)} with ground-truth rows · hub on :{base_port}</p>"
        f"<div class=scroll><table><thead><tr><th>Funder</th><th>Snapshot</th><th>Pages</th>"
        f"<th>Docs</th><th>Files</th><th>CSS/JS</th><th>Ground truth</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        f"<p class=note style='margin-top:1.5rem'>Each snapshot runs on its own port because the "
        f"mirrored pages use root-absolute links and asset paths; a shared origin would break them. "
        f"Links back to the funder's own domain are rewritten to that port as the page is served, so "
        f"clicking can never land you on the live site — an uncaptured link gets a \"not in this "
        f"snapshot\" page instead. Assets (css/js/images) are only rewritten when they are in the "
        f"snapshot; where the crawl captured none, the browser loads them live so the page is still "
        f"readable, and the CSS/JS column says which snapshots those are. Bytes on disk are untouched, "
        f"and the eval-time server rewrites nothing.</p>"
    )
    return page("msc-code snapshots", body)


def gt_view(site: dict) -> bytes:
    rows = site["gt_rows"]
    if not rows:
        return page(site["funder"], "<a class=back href='/'>← all snapshots</a><p>No labelled rows yet.</p>")
    cols = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body_rows = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c) or ""
            if c == "title":
                target = local_url(site, r.get("url", ""), r.get("source_page", ""))
                cell = (
                    f"<a href='{html.escape(target)}' target=_blank>{html.escape(v)}</a>"
                    if target
                    else html.escape(v)
                )
                cells.append(f"<td class=wrap>{cell}</td>")
            elif c in ("url", "source_page"):
                target = local_url(site, r.get("url", ""), r.get("source_page", ""))
                cells.append(
                    f"<td class='wrap mono'>"
                    + (f"<a href='{html.escape(target)}' target=_blank>{html.escape(v)}</a>" if target else html.escape(v))
                    + "</td>"
                )
            elif c in ("notes", "summary", "eligibility_summary", "amount_raw"):
                cells.append(f"<td class=wrap>{html.escape(v)}</td>")
            else:
                cells.append(f"<td>{html.escape(v)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    body = (
        f"<a class=back href='/'>← all snapshots</a>"
        f"<h1>{html.escape(site['funder'])} — ground truth</h1>"
        f"<p class=sub>{len(rows)} rows · snapshot {site['date']} · "
        f"<span class=mono>{html.escape(site['gt_path'].name)}</span> · "
        f"site on <a href='http://localhost:{site['port']}{site.get('landing', '/')}' "
        f"target=_blank>:{site['port']}</a></p>"
        f"<div class=scroll><table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )
    return page(f"{site['funder']} ground truth", body)


def pages_view(site: dict) -> bytes:
    entries = [e for e in site["manifest"]["files"] if e.get("kind") == "page"]
    docs = [e for e in site["manifest"]["files"] if e.get("kind") == "document"]
    def rows_for(items):
        out = []
        for e in sorted(items, key=lambda e: e["url"]):
            target = local_url(site, e["url"], "")
            out.append(
                f"<tr><td class=wrap><a href='{html.escape(target)}' target=_blank>"
                f"{html.escape(e.get('title') or e['url'])}</a></td>"
                f"<td class='wrap mono'>{html.escape(e['url'])}</td>"
                f"<td class='num nowrap'>{e.get('bytes', 0):,}</td></tr>"
            )
        return "".join(out)
    body = (
        f"<a class=back href='/'>← all snapshots</a>"
        f"<h1>{html.escape(site['funder'])} — captured pages</h1>"
        f"<p class=sub>{len(entries)} pages · {len(docs)} documents · snapshot {site['date']}</p>"
        f"<div class=scroll><table><thead><tr><th>Title</th><th>URL</th><th>Bytes</th></tr></thead>"
        f"<tbody>{rows_for(entries)}</tbody></table></div>"
    )
    if docs:
        body += (
            f"<h2>Documents ({len(docs)})</h2><div class=scroll><table><thead><tr>"
            f"<th>Title</th><th>URL</th><th>Bytes</th></tr></thead>"
            f"<tbody>{rows_for(docs)}</tbody></table></div>"
        )
    return page(f"{site['funder']} pages", body)


def make_browse_handler(site: dict, hub_port: int, misses: Counter, rewrite: bool, strict: bool):
    """Serve one snapshot, resolving live URLs to mirrored files (as ``serve.py`` does),
    then fixing up absolute links so clicking stays on localhost."""
    lookup = build_lookup(site["dir"], site["manifest"])
    hosts = origins_for(site["manifest"])
    port = site["port"]

    # resolve() falls back to a linear scan over the lookup, and rewriting asks about
    # every URL on every page served, so memoise; the answer cannot change at runtime.
    seen: dict[tuple[str, str], bool] = {}

    def captured(path: str, query: str) -> bool:
        key = (path, query)
        if key not in seen:
            seen[key] = resolve(lookup, path, query) is not None
        return seen[key]

    site["captured"] = captured
    site["landing"] = landing_path(site, captured)
    site["re_plain"] = _host_re(hosts, escaped=False)
    site["re_escaped"] = _host_re(hosts, escaped=True)

    class Browse(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            target = resolve(lookup, parsed.path, parsed.query)
            if target is None:
                misses[parsed.path] += 1
                self._fail(404, parsed.path)
                return

            body = target.read_bytes()
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            is_html = ctype in ("text/html", "application/xhtml+xml")
            live = 0
            if rewrite and (is_html or ctype == "text/css"):
                body, live = rewrite_html(body, site, is_css=not is_html, strict=strict)
            if is_html:
                body = inject(
                    body,
                    guard(origins_for(site["manifest"]))
                    + badge(site["funder"], site["date"], port, hub_port, live),
                )
            if ctype.startswith("text/") or ctype == "application/xhtml+xml":
                ctype += "; charset=utf-8"

            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def _fail(self, code: int, path: str) -> None:
            body = page(
                "not in this snapshot",
                f"<h1>Not in this snapshot</h1>"
                f"<p class=sub><span class=mono>{html.escape(path)}</span> was not captured for "
                f"<strong>{html.escape(site['funder'])}</strong> ({site['date']}).</p>"
                f"<p class=note>The crawl has a boundary; this is what it looks like from inside. "
                f"<a href='http://localhost:{hub_port}/pages/{site['funder']}'>See every captured page</a> "
                f"or go back to the <a href='http://localhost:{hub_port}/'>hub</a>.</p>",
            )
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            pass

    return Browse


def make_hub_handler(sites: list[dict], base_port: int):
    by_name = {s["funder"]: s for s in sites}

    class Hub(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/":
                self._send(hub_index(sites, base_port))
            elif path.startswith("/gt/") and path[4:] in by_name:
                self._send(gt_view(by_name[path[4:]]))
            elif path.startswith("/pages/") and path[7:] in by_name:
                self._send(pages_view(by_name[path[7:]]))
            else:
                self._send(page("not found", "<a class=back href='/'>← all snapshots</a><p>404</p>"), 404)

        def _send(self, body: bytes, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            pass

    return Hub


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="data/snapshots")
    p.add_argument("--ground-truth", default="data/ground-truth")
    p.add_argument("--port", type=int, default=8100, help="hub port; funders take port+1, +2, ...")
    p.add_argument(
        "--no-rewrite",
        action="store_true",
        help="serve the captured bytes verbatim; absolute links will leave localhost",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="never fall back to the live web for uncaptured assets (pages may render bare)",
    )
    args = p.parse_args()

    snapshots_root = Path(args.data).resolve()
    sites = discover(snapshots_root, Path(args.ground_truth).resolve())
    if not sites:
        print(f"error: no snapshots under {snapshots_root}")
        return 1

    servers = []
    misses: dict[str, Counter] = {}
    for i, s in enumerate(sites, start=1):
        s["port"] = args.port + i
        files = s["manifest"]["files"]
        s["n_files"] = len(files)
        s["n_pages"] = sum(1 for f in files if f.get("kind") == "page")
        s["n_docs"] = sum(1 for f in files if f.get("kind") == "document")
        s["host"] = urlparse(files[0]["url"]).netloc if files else ""
        s["n_style"] = sum(
            1 for f in files if urlparse(f["url"]).path.lower().endswith((".css", ".js"))
        )
        s["style_cell"] = (
            f"<span class='pill ok'>{s['n_style']}</span>"
            if s["n_style"]
            else "<span class='pill pend' title='none captured — pages fall back to live styling'>none</span>"
        )

        counter: Counter = Counter()
        misses[s["funder"]] = counter
        handler = make_browse_handler(
            s, args.port, counter, rewrite=not args.no_rewrite, strict=args.strict
        )
        srv = ThreadingHTTPServer(("127.0.0.1", s["port"]), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        gt = f"{len(s['gt_rows'])} gt rows" if s["gt_rows"] else "no gt rows"
        print(f"  :{s['port']}  {s['funder']:<28} {s['date']}  {s['n_pages']:>4} pages  {gt}")

    hub = ThreadingHTTPServer(("127.0.0.1", args.port), make_hub_handler(sites, args.port))
    servers.append(hub)
    print(f"\n  hub: http://localhost:{args.port}/    (ctrl-c to stop all)")
    try:
        hub.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for srv in servers:
            srv.server_close()
        hit = {k: v for k, v in misses.items() if v}
        if hit:
            print("\nURLs requested but not in a snapshot:")
            for funder, counter in hit.items():
                for url, n in counter.most_common(10):
                    print(f"  {funder}: {n:>3}x {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
