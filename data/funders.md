# Funders and tune/validation split

`data/funders.json` is the registry the code reads. `scripts/build_funders_json.py`
builds it from the split lists below and from each snapshot's files, so correct this
file first and then rebuild the JSON.

## Current set

24 UK funders: 16 tune and 8 validation. The validation funders were not looked at
during tuning. Wellbeing of Women was in the validation split until 2026-09-13, when it
was withdrawn because none of its rows were in scope (see `eval/preregistration.md` and
`data/ground-truth/withdrawn/`).

The ground truth for the 24 funders has 292 candidate rows, 167 of them in scope. The
grid has 120 cells per funder, so the 16 tune funders give 1,920 runs. The validation
split has 120 x 8 = 960 possible runs, and the lock file picks the single-shot subset.

- **Tune (16):** leverhulme, nuffield, wolfson, henry-moore, paul-mellon, carnegie-trust,
  garfield-weston, british-ecological-society, rsc, physiological-society,
  society-antiquaries, sasakawa, nuffield-farming, cara, british-academy,
  genetics-society.
- **Validation (8, disjoint, held out):** heritage-fund, fulbright, royal-hist-soc,
  society-of-authors, hakluyt, pilgrim-trust, pathological-society,
  prs-foundation.

## Which ground-truth files count

The loader reads only `<slug>_<YYYY-MM-DD>.csv` files directly inside
`data/ground-truth/`. Two other CSVs in that folder are not ground truth:
`ground-truth_all-funders (rsc).csv` is a summary table, and
`ground-truth_all-funders(british-ecological-society).csv` is an old export of the BES
rows from before they were corrected. Its header looks like a real ground-truth header,
so it must not be loaded. Subfolders are not read, which keeps out
`withdrawn/wellbeing-of-women_2026-08-21.csv` and the two deferred funders in
`deferred-batch3/` (raeng and ihr). Moving any of these into the main folder changes
the row counts, the tune split and the grid size together.

## How the funders were chosen

Each funder had to be UK-based, publish its grant pages openly, serve them as plain
HTML (the snapshot tool uses `wget`, so sites built only in JavaScript come back empty)
and allow crawling without a firewall block on the research user agent. Every candidate
was probed before it was picked.

All snapshots use depth 3 and `--assets none`. From the 2026-08-16 captures onwards a
reject filter skips query strings, `wp-json`/`oembed`/`feed`, `wp-content/cache` and
static media, style, font and archive files, with a 0.5s wait and a 2m rate limit.
Without it several WordPress sites crawled for hours through asset and API URLs; it
removes no pages or PDFs. Leverhulme (2026-08-09) and Henry Moore (2026-08-14) were
captured before the filter with a 1s / 500k throttle, and neither site had the problem.
Large sites were crawled from their funding section with `--no-parent`, small ones from
the root. `--no-parent` only bounds a crawl to the directory of the entry URL, so
batch 3 entries are directory-shaped section paths. Exact parameters are in each
snapshot's `meta/snapshot.json`.

In the Source column, "catalogue" means the funder came from a list of UK grant
databases and "discovered" means it was found by search to widen the mix of funder types.
The ground truth for every funder was labelled from its snapshot.

## Batch 1: funders 1 to 11

| #  | Funder | slug | Type | Split | Source | Entry URL | Crawl notes | Snapshot date |
|----|--------|------|------|-------|--------|-----------|-------------|---------------|
| 1  | The Leverhulme Trust | `leverhulme` | charity trust | tune | catalogue | https://www.leverhulme.ac.uk/ | root, depth 3 | 2026-08-09 |
| 2  | Nuffield Foundation | `nuffield` | charity | tune | catalogue | https://www.nuffieldfoundation.org/funding | funding section | 2026-08-16 |
| 3  | Wolfson Foundation | `wolfson` | charity | tune | catalogue | https://www.wolfson.org.uk/funding/ | funding section | 2026-08-16 |
| 4  | Henry Moore Foundation | `henry-moore` | niche trust (sculpture/visual arts) | tune | catalogue | https://henry-moore.org/what-we-do/grants-fellowships/ | grants section (canonical non-www host) | 2026-08-14 |
| 5  | Paul Mellon Centre for Studies in British Art | `paul-mellon` | niche trust (art history) | tune | catalogue | https://www.paul-mellon-centre.ac.uk/ | root, depth 3 | 2026-08-16 |
| 6  | Carnegie Trust for the Universities of Scotland | `carnegie-trust` | niche trust | tune | discovered | https://carnegie-trust.org/ | root (canonical non-www host) | 2026-08-16 |
| 7  | Garfield Weston Foundation | `garfield-weston` | charity | tune | discovered | https://garfieldweston.org/ | root, depth 3 | 2026-08-16 |
| 8  | British Ecological Society | `british-ecological-society` | professional body | tune | discovered | https://www.britishecologicalsociety.org/funding/ | funding section (redirects to /section-type/funding/) | 2026-08-16 |
| 9  | National Lottery Heritage Fund | `heritage-fund` | heritage funder | validation | catalogue | https://www.heritagefund.org.uk/funding | funding section, `--no-parent` (large site) | 2026-08-16 |
| 10 | US-UK Fulbright Commission | `fulbright` | exchange / professional body | validation | catalogue | https://fulbright.org.uk/ | root (canonical non-www host; www entry broke recursion) | 2026-08-16 |
| 11 | Royal Historical Society | `royal-hist-soc` | professional body (humanities) | validation | discovered | https://royalhistsoc.org/grants/ | grants section | 2026-08-16 |

Probed and left out: Wellcome Trust and Art Fund (JavaScript-only), British Council
(connection refused), Royal Society and Royal Society of Edinburgh (403). British
Academy also returned 403 at this point but was captured cleanly in batch 3.

## Batch 2: funders 12 to 21

This batch widened the mix of types: professional and learned bodies in chemistry,
physiology and literature, a women's-health charity, a UK-Japan exchange trust, a
heritage and social-welfare trust, an agricultural scholarship trust and a charity for
at-risk academics. Snapshots are dated 2026-08-21.

| #  | Funder | slug | Type | Split | Source | Entry URL | Crawl notes | Snapshot date |
|----|--------|------|------|-------|--------|-----------|-------------|---------------|
| 12 | Royal Society of Chemistry | `rsc` | professional body (chemistry) | tune | discovered | https://www.rsc.org/funding-and-support/funding | funding section, `--no-parent` (large corporate site) | 2026-08-21 |
| 13 | The Physiological Society | `physiological-society` | professional body (physiology) | tune | discovered | https://www.physoc.org/grants-and-prizes/grants/ | grants section, `--no-parent` (root `/grants/` redirects here) | 2026-08-21 |
| 14 | Society of Authors | `society-of-authors` | professional body (literature / authors) | validation | discovered | https://societyofauthors.org/grants/ | grants section, `--no-parent` | 2026-08-21 |
| 15 | Wellbeing of Women | `wellbeing-of-women` | medical charity (women's health) | withdrawn (was validation; 0 in-scope rows) | discovered | https://www.wellbeingofwomen.org.uk/what-we-do/research/ | research/grants section, `--no-parent` | 2026-08-21 |
| 16 | Society of Antiquaries of London | `society-antiquaries` | learned society (archaeology / heritage) | tune | discovered | https://www.sal.org.uk/what-we-do/grants/ | grants section, `--no-parent` (root `/grants/` redirects here) | 2026-08-21 |
| 17 | Hakluyt Society | `hakluyt` | learned society (history of exploration / geography) | validation | discovered | https://www.hakluyt.com/ | root, depth 3 (small learned-society site) | 2026-08-21 |
| 18 | The Pilgrim Trust | `pilgrim-trust` | charity (heritage & social welfare) | validation | discovered | https://www.thepilgrimtrust.org.uk/ | root, depth 3 | 2026-08-21 |
| 19 | Great Britain Sasakawa Foundation | `sasakawa` | niche trust (UK–Japan; incl. medicine/health & research) | tune | discovered | https://www.gbsf.org.uk/grants/ | grants section, `--no-parent` | 2026-08-21 |
| 20 | Nuffield Farming Scholarships Trust | `nuffield-farming` | niche trust (agriculture / food / rural: study scholarships) | tune | discovered | https://www.nuffieldscholar.org/scholarships/what-nuffield-farming-scholarship | scholarships section, `--no-parent` (root wanders site-wide) | 2026-08-21 |
| 21 | Cara (Council for At-Risk Academics) | `cara` | charity (at-risk / refugee academics) | tune | discovered | https://www.cara.ngo/what-we-do/ | what-we-do section, `--no-parent` | 2026-08-21 |

Probed and left out:

- Royal Geographical Society: its `robots.txt` is malformed, so `wget` could not crawl
  past the entry page.
- Churchill Fellowship, Daiwa Anglo-Japanese Foundation: a root crawl spread over the
  whole site and no bounded grants section was found.
- Iolanthe Midwifery Trust, British Institute at Ankara, Institute of Physics: 403.
- Royal Academy of Engineering: a root crawl was mostly Fellow profile pages. A bounded
  entry was found in batch 3.
- Biochemical Society (404), Economic History Society and Bibliographical Society
  (unstable entry pages), Marc Fitch Fund (a single page).

## Batch 3: funders 22 to 25

This batch added a national academy, a medical professional body, a music funder and a
life-sciences society. Snapshots are dated 2026-08-28. pathological-society went into
validation to replace wellbeing-of-women with a medical funder that has in-scope rows.

| #  | Funder | slug | Type | Split | Source | Entry URL | Crawl notes | Snapshot date |
|----|--------|------|------|-------|--------|-----------|-------------|---------------|
| 22 | The British Academy | `british-academy` | national academy (humanities & social sciences) | tune | discovered (re-probed) | https://www.thebritishacademy.ac.uk/funding/ | funding section, `--no-parent`; returned 200 on re-probe after a 403 in batch 1 | 2026-08-28 |
| 23 | The Pathological Society of Great Britain and Ireland | `pathological-society` | professional body (pathology / medicine) | validation | discovered | https://www.pathsoc.org/grants_lectures_awards/default.aspx | grants section, `--no-parent`; replaces wellbeing-of-women, which had 0 in-scope rows | 2026-08-28 |
| 24 | PRS Foundation | `prs-foundation` | arts funder (music) | validation | discovered | https://prsfoundation.com/funding-support/ | funding section, `--no-parent` | 2026-08-28 |
| 25 | The Genetics Society | `genetics-society` | learned society (genetics) | tune | discovered | https://genetics.org.uk/grants/ | grants section, `--no-parent` | 2026-08-28 |

The Royal Academy of Engineering (`raeng`, 40 rows, 15 in scope) and the Institute of
Historical Research (`ihr`, 27 rows, 5 in scope) were captured and labelled the same
way, then held back to keep the in-scope total near 150 (it would otherwise be 187).
Both would have been tune funders. Their files are in `data/ground-truth/deferred-batch3/`.
Leaving out `raeng` means the set has no engineering funder.

About 70 other candidates were probed and left out because of a 403 block, an
unbounded grants section that made the crawl spread over the whole site, too few pages,
or no stable grants page (404).
