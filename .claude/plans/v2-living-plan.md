# Grimoire v2 — Living Plan

> Exported from the living plan doc on 2026-10-21. The doc is the working copy; this file is a snapshot for the repo.

## Overview

### How to use this plan

This plan replaces every plan made before 2026-10-14, including the post-v1 ranked backlog; v1 (Phases 0–7) is the shipped baseline. It is split into four regions, each in its own tab and each built around one interface contract, so the regions can be developed in parallel:

- **R1 · Harvester interface** — contract C1 only; Harvester itself is built elsewhere.
- **R2 · Storage, RAG, cartography** — Grimoire's core, up to contract C2, the Retrieval API.
- **R3 · LLM inquiry** — contract C3, one port to any LLM, direct or through a gateway.
- **R4 · Runners** — contract C4, pushing batch work to local, HPC or cloud runners.

This Overview holds what spans regions: goals, scope, architecture, the contracts summary and shared conventions, decisions, roadmap, cross-region questions and risks.

- **Decisions** change only by adding a row that supersedes the old one.
- **Acceptance criteria** are oracle checks or conformance suites, written before the code.
- **Schema changes** are reviewed table by table with Rodrigo as they are written.
- Sections marked *Exploration* are hypotheses to test, not commitments.

### Context and goals

Grimoire v2 turns a personal literature server (15.5k items) into the information layer for an institutional research-library RAG over roughly 0.5–1M papers, while still running on a personal NAS from the same code.

**Where v1 stands (2026-10-14):** Phases 0–7 done, 309 tests passing, 15,547 items and 42 GB of files migrated from Zotero. The heavy compute has not run on the real library yet: 0 chunks, 0 embeddings, 2 GROBID TEI artifacts.

**Two deployments, one codebase:**

|  | Personal (NAS) | Institutional (intranet server) |
| --- | --- | --- |
| Owner | Rodrigo | the hosting institution's library |
| Corpus | \~15k curated items (Zotero) | journal archives via publisher TDM APIs and SFTP pushes |
| Scale | \~15k items, \~1M chunks | \~0.5–1M papers, \~30–50M chunks (estimate) |
| Users | one | all staff on the intranet, no access control |
| LLM access | Anthropic API | an LLM gateway (e.g. LiteLLM) to hosted and internal models |
| Batch compute | local, or rented GPU | institutional HPC (Slurm), optionally AWS |
| Clients | Claude Code | Claude Code, other MCP clients, a future chatbot |

Neither deployment queries the other.

**v2 must deliver:**

1. Ingest at journal-archive scale from Harvester payloads, idempotently.
2. Store literature at every granularity (full files through chunks) with full provenance from any retrieved unit back to the source file and page.
3. Retrieval in three modes (raw, filtered, curated) over MCP, web UI and OPDS.
4. Batch compute that runs unchanged on local, Slurm or AWS-style runners.
5. A navigable map of the corpus (cartography layer) for routing and big-picture questions — *Exploration*.
6. The v1 quality bars on dedup, metadata and recall, held at the new scale.

### Scope and non-goals

Grimoire and Harvester are Rodrigo's personal open-source projects (MIT). Each site runs its own deployment; site-specific configuration and secrets stay with that site, not in the repo.

**In Grimoire core:** everything generic — identification, dedup, parsing, chunking, embedding, indexing, storage, provenance, retrieval modes, the runner interface, MCP, web UI, OPDS, and extension points (payload sources, artifact kinds, runners, LLM provider).

**Outside the repo (per site):** deployment configuration and secrets, publisher credentials, query-layer clients such as chatbots, models trained on a site's resources, and HPC wrappers specific to one cluster.

**Non-goals for v2:**

- Access control or multi-tenancy. Target sites serve non-restricted content inside one network; the NAS has one user.
- Answer generation inside Grimoire. Grimoire returns evidence; the client (Claude or a chatbot) writes answers. Exception: map summaries for big-picture questions.
- Federation between deployments.
- Training an answering LLM. Training effort goes to retrieval (embeddings, reranker) and parsing.
- A dedicated graph database or vector database, unless Postgres measurably fails a performance oracle.
- Per-chunk LLM extraction across a large corpus (full Microsoft-style GraphRAG indexing).
- Designing the query layer. Grimoire's scope ends at the Retrieval API (C2); Claude Code is the first consumer, and other clients are built later against the same contract.

**Baseline:** tag `80147e9` as the v1 baseline.

### Architecture overview

Grimoire is the information layer: R2 is its core, R3 and R4 are its ports to models and to compute, and the regions touch only through contracts C1–C4.

*Diagram: v2 architecture · four regions, four contracts (see the living doc).*

Harvester only talks to publishers and to C1; consumers only talk to C2. On the NAS there is no Harvester: local ingest paths build C1 manifests internally, and the runner and LLM provider are local choices.

### Interface contracts at a glance

Four contracts let the four regions be built in parallel; each is a versioned spec with fixtures and a mock, so either side can be developed and tested without the other.

| Contract | Between | Transport | Spec lives in | Defined in |
| --- | --- | --- | --- | --- |
| C1 · Payload API | Harvester → Grimoire | HTTPS + JSON; Grimoire fetches files over HTTPS | `contracts/payload/v1/` (OpenAPI + JSON Schema) | R1 tab |
| C2 · Retrieval API | Grimoire → any consumer (Claude Code first) | HTTPS + JSON; MCP tools are a thin adapter over it | `contracts/retrieval/v2/` (OpenAPI) | R2 tab |
| C3 · LLM port | Grimoire code → LLM providers or gateways | in-process Python interface; adapters speak Anthropic or OpenAI-compatible HTTP | `src/grimoire/llm/port.py` + `contracts/llm/` (task schemas) | R3 tab |
| C4 · Job contract | Grimoire dispatcher → runners → job containers | JobSpec / result JSON; OCI images run by Docker or Apptainer | `contracts/jobs/v1/` (JSON Schema) | R4 tab |

The query layer (Claude Code, later chatbots and other clients) is out of scope beyond C2. Claude Code is the first consumer, through the MCP adapter.

### Cross-cutting conventions

Every contract follows the same rules, so a developer who learns one can read the others.

| Topic | Convention |
| --- | --- |
| File identity | `sha256` of the bytes, lowercase hex; the same hash names the file in Grimoire's store, in manifests and in job inputs |
| Record identity | Grimoire IDs are UUIDv7 (time-ordered); external IDs (DOI, PII, OpenAlex, publisher IDs) travel in an `identifiers` object, never as primary keys |
| Idempotency | every write carries a client-chosen key (`payload_id`, `job_id`, `request_id`); repeating it with the same body returns the original result, with a different body returns `409` |
| Time | RFC 3339 UTC timestamps (`2026-10-14T18:03:00Z`) |
| Errors | one envelope everywhere: `{"error": {"code": "manifest_invalid", "message": "…", "details": {}, "request_id": "…"}}`; codes are stable strings, messages are not |
| Versioning | major version in the path or `*_version` field; within a major version, changes are additive only (new optional fields, new enum values that consumers must tolerate) |
| Auth | service-to-service bearer tokens, even on the intranet; read-only Retrieval API may run without auth inside a trusted network |
| Specs | OpenAPI 3.1 for HTTP, JSON Schema 2020-12 for documents, both in `contracts/` in the Grimoire repo |
| Testing | each contract ships fixtures, a mock of the other side, and a conformance suite that both sides run in CI |

### Decision log

Seventeen decisions are settled, nine are proposals this plan argues for, and D10 is superseded by D16. D1–D20 date from the 2026-10-14 design discussion and D21–D27 from 2026-10-20; new rows go on top.

| # | Decision | Why | Status |
| --- | --- | --- | --- |
| D27 | How the runner reaches the HPC and which API formats the LiteLLM gateway exposes are settled by probes at the deploying site; site-specific adapters and gateway config are built there | both depend on the site's network and services | Settled |
| D26 | The chunk embedding model and its dimension are chosen on the retrieval eval set before any large backfill | re-embedding 30–50M chunks is the costliest mistake available | Settled |
| D25 | Provenance granularity: paragraph or chunk → section or chapter → file (sha256) → paper or book; page range too when the source has pages | any result leads to where it came from; XML has sections but no pages | Settled |
| D24 | Each publisher is a small Python adapter over a shared base; API key and institutional token live only in config; Elsevier first; a config-only generic adapter is decided after the second publisher | one publisher cannot show what is common | Proposed |
| D23 | Harvester is one codebase with roles: all (beside Grimoire) and edge (standalone in a DMZ, receiving SFTP drops or running a listener; the inner Harvester always fetches from it, after polling or a notification) | publishers deliver into a DMZ; the same code runs in both places | Settled |
| D22 | Docker Compose is the deployment format at every site | TrueNAS runs it, and so does any Docker or Podman host | Settled |
| D21 | One repo, two packages and two images (grimoire, harvester); contracts/ shared; a test forbids Harvester from importing Grimoire code | one owner; C1 and its kit change in the same commit as both sides; still splittable later | Proposed |
| D20 | Every plan made before 2026-10-14 (post-v1 ranked backlog, next-session plan) is superseded; each old issue gets an explicit disposition | the vision and requirements changed | Settled |
| D19 | Grimoire's scope ends at the Retrieval API (C2); the query layer is designed later; Claude Code is the first consumer | Claude Code exists and is the primary target | Settled |
| D18 | The plan is split into four regions, each built around a versioned contract (C1–C4) with fixtures, mocks and a conformance suite | regions, including Harvester elsewhere, can be built in parallel | Settled |
| D17 | The Retrieval API is HTTP + JSON; MCP is a thin adapter over it | one contract for Claude Code, chatbots and any UI | Proposed |
| D16 | LLM access goes through Grimoire's own port with two adapters (Anthropic Messages, OpenAI-compatible); no LiteLLM library dependency. Supersedes D10 | works directly or through any gateway, including LiteLLM | Proposed |
| D1 | Grimoire and Harvester are Rodrigo's personal projects; each site runs its own deployment | keeps the project independent of any one deployment | Settled |
| D2 | Independent deployments (personal NAS, institutional server) share one codebase; neither queries the other | sites have separate networks and corpora | Settled |
| D3 | Three layers: Harvester (ingestion) → Grimoire (information) → Claude / chat clients (query) | separates acquisition, knowledge and answering | Settled |
| D4 | Harvester is transient storage; it notifies Grimoire, and Grimoire is the permanent store of files and RAG units | one source of truth; Harvester stays simple | Settled |
| D5 | Full provenance: every retrievable unit resolves to a stored file and page range | users must be able to get the source paper | Settled |
| D6 | Batch compute is push-based: Grimoire's queue → dispatcher → configurable runner (local, Slurm, AWS Batch) | pull workers would force an always-on AWS instance | Settled |
| D7 | Postgres replaces SQLite on both deployments (approved 2026-10-14) | invariants become structural; concurrent writers; networked layers | Settled |
| D8 | Postgres extensions: pgvector for vectors, ParadeDB pg\_search for BM25 | one store for relational, vector and keyword search | Proposed |
| D9 | No graph database: graph as edge tables; graph analytics run offline as batch jobs | live queries need 1–2 hops; heavy work is offline | Proposed |
| D10 | LiteLLM is Grimoire's only LLM client | matched an existing gateway | Superseded |
| D11 | Retrieval modes raw / filtered / curated; curated returns evidence, not answers | avoids paying for two LLM passes | Proposed |
| D12 | Prefer publisher XML (MathML) over PDF parsing; keep GROBID for metadata and references; add a body extractor for equations | exact equations where XML exists | Proposed |
| D13 | Two-stage retrieval: papers first, then chunks within candidate papers | avoids one ANN index over \~50M chunks | Proposed |
| D14 | The server stores the PDFs; serving PDFs, excerpts and links follows each source's TDM terms | negotiated with publishers | Settled |
| D15 | Start with a small VM; scale up as utility is proven | cost follows proven use | Settled |

### Deployment profiles

The same images and schema run everywhere; a profile is configuration only (runner, LLM endpoint, storage URIs, which features are on).

| Setting | NAS (personal) | Institutional server | Institutional HPC (batch only) |
| --- | --- | --- | --- |
| Services | Grimoire, Postgres, GROBID, translation-server | same + Harvester, collector and dispatcher | job images under Apptainer |
| Runner | `local` (or `aws-batch` for a one-off backfill) | `slurm`, optionally `aws-batch` | — |
| LLM | Anthropic API via LiteLLM | LLM gateway (e.g. LiteLLM) to hosted and internal models | internal models for batch tasks, if allowed |
| Files | ZFS dataset, `files/` default recordsize, Postgres on a `recordsize=16K` dataset | TB-scale volume for PDFs; SSD for Postgres | staging only |
| Starting size | existing X9SCM-F box | \~8 vCPU, 32 GB RAM, a few hundred GB SSD + bulk storage for PDFs (first guess) | per job |
| Auth | none, LAN only | none inside the site network | cluster accounts |

**Packaging (D21–D23):** one repo builds two images, `grimoire` and `harvester`, and each site runs a Docker Compose file. The NAS runs Grimoire, Postgres and GROBID, with Harvester optional. An institutional server adds Harvester in the `all` role. A DMZ host, if needed, runs Harvester alone in the `edge` role. Publisher keys, tokens and site settings live in each site's config, never in the repo.

### Roadmap

Each region runs its own lane of milestones, and each lane starts by freezing its contract draft so the others can code against it. Gates for every milestone live in the region tabs. This replaces the earlier nine-phase roadmap.

*Diagram: v2 roadmap · four region lanes, 24 milestones (see the living doc).*

**Where lanes wait on each other:**

- H3 (payload ingest) needs S1 (Postgres core).
- S3 (pipeline as tasks) needs J2 (local runner) and J3 (first job images).
- S5 (institutional pilot) needs H5 (real Harvester) and J4 (Slurm runner).
- S6 (`curated` mode) needs L5 (query tasks); S7 (cartography) needs L6 (batch summaries).
- L3 (existing LLM tasks moved) needs S1, because the call log lives in Postgres.

The first contract drafts (H1, J1, L1) and S1 have no dependencies and can start together.

### Pre-session backlog: disposition

Every item planned before 2026-10-14 is either folded into a region milestone, deferred, or closed; nothing carries over unexamined (D20). No GitHub issue has been changed yet.

| Old item | What it was | Disposition | Lands in |
| --- | --- | --- | --- |
| [#1](https://github.com/deOliveira-R/grimoire/issues/1) | KOReader OPDS roundtrip oracle | kept as a regression check for OPDS on the NAS; no longer gates a release | R2 |
| [#2](https://github.com/deOliveira-R/grimoire/issues/2) | MCP research walkthrough oracle | replaced by the S2 gate: the walkthrough through the new Retrieval API | R2 · S2 |
| [#3](https://github.com/deOliveira-R/grimoire/issues/3) | 20-query recall@10 oracle | becomes the seed of the retrieval eval set | R2 · S2 |
| [#6](https://github.com/deOliveira-R/grimoire/issues/6), [#7](https://github.com/deOliveira-R/grimoire/issues/7), [#9](https://github.com/deOliveira-R/grimoire/issues/9), [#10](https://github.com/deOliveira-R/grimoire/issues/10) | web UI: author facet, keyboard nav, split pane, multi-select | deferred: UI work sits outside the four regions for now | — |
| [#12](https://github.com/deOliveira-R/grimoire/issues/12) | upload and URL-submit forms | deferred; when built, it becomes a local C1 payload source | R1 · H4 later |
| [#13](https://github.com/deOliveira-R/grimoire/issues/13) | v2 umbrella (CSL styles, annotations, tags, author disambiguation) | close as superseded by this plan; re-file items individually if still wanted | — |
| [#14](https://github.com/deOliveira-R/grimoire/issues/14) | GROBID fallback for books without a TOC | folded into the extraction tasks, low priority | R2 · S4 |
| [#15](https://github.com/deOliveira-R/grimoire/issues/15) | tier-4 dedup review queue | kept, after the v1 library migration | R2 · after S1 |
| [#16](https://github.com/deOliveira-R/grimoire/issues/16) | citation graph from TEI | broadened: citations from publisher XML, OpenAlex and TEI feed cartography | R2 · S7 |
| [#18](https://github.com/deOliveira-R/grimoire/issues/18) | OCR fallback (migration 005, local subprocess) | design superseded: OCR is an extraction task run through runners, chosen on the old-page set | R2 · S4, R4 |
| [#19](https://github.com/deOliveira-R/grimoire/issues/19) | fuzzy matching for DOI-less references | kept as a cartography input | R2 · S7 |
| Old plan, session 2.1 | GROBID backfill of the 15k library on SQLite | replaced: runs as the first real job after S1 and J3 | R2 · S3 |
| Old plan, sessions 3 and 5 | HTMX UI pair; OCR via migration 005 | superseded (see #7, #9, #18) | — |
| v1.0 tag | blocked on #1–#3 | dropped as a goal; tag `80147e9` as the v1 baseline instead | Overview |

The useful parts of the old plans (OCR measurements, citation-build rules, GROBID sizing, v1's named traps, v1's MCP tools) are now in R2, R4 and the risks below. The old backlog file was then removed from the repo; it stays in git history at `8e101a1`.

### Open questions

Questions that belong to one region now live in that region's tab; this table keeps only those that span regions or concern the project as a whole.

| # | Question | Owner | Blocks | Status |
| --- | --- | --- | --- | --- |
| Q1 | Approve the move to Postgres (D7) | Rodrigo | S1 | Answered |
| Q3 | Apply the backlog disposition to GitHub (close #13, relabel the rest by region)? | Rodrigo | — | Open |
| Q4 | How does a deploying site consume Grimoire? Answered: container images plus a per-site compose file and config (D21, D22) | Rodrigo | H5, S5 | Answered |
| Q7 | Do the TDM agreements permit training models on the content (PDF→XML model)? | Rodrigo | R2 exploration | Open |

### Risks and mitigations

The largest risks are scope creep from institutional scale pulling the personal deployment along, and quality loss hidden behind volume.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Postgres keyword ranking weaker than FTS5 BM25 | recall drops after migration | ParadeDB pg\_search; v1 recall oracle must not regress (≥ 90% hybrid recall@10) |
| Institution-driven features the NAS doesn't need | personal deployment becomes heavy | profiles are config only; every service optional except Postgres |
| VLM OCR errors look plausible (dropped subscript, wrong sign) | silent equation errors | keep page images; cross-check against raw-body XML or old OCR layer; flag disagreement |
| `curated` mode drops relevant evidence | answers miss key papers | report filtered-out IDs; per-mode recall oracle |
| Cartography summaries drift from sources | misleading big-picture answers | `derivations` provenance; reader panel review; exit criteria before rollout |
| HPC network rules block job I/O | runner stalls | answer R4's network questions before J4; \`posix\` staging fallback |
| Publisher XML coverage lower than assumed | more PDF parsing and OCR cost | measure coverage per journal and year during the pilot |
| Author-key collisions ("smith,j") grow with the corpus | tier-4 dedup's shared-author test weakens; false merges at 1M items | use ORCID and OpenAlex author IDs where present; re-run the dedup oracle on pairs sampled from the large corpus, not only the 15k library |
| Metadata API rate limits (Crossref, OpenAlex) at 1M lookups | identification stalls the pipeline | publisher metadata from C1 first; polite-pool headers; cache every response; an OpenAlex snapshot for bulk lookups |

## R1 · Harvester interface

Harvester is a separate project developed elsewhere; this tab defines only contract C1, the Payload API through which Harvester hands literature to Grimoire, plus the kit both sides use to test against it.

### Boundary

| Concern | Harvester | Grimoire |
| --- | --- | --- |
| Publisher credentials, TDM API clients, SFTP drop | owns | never sees |
| Rate limits, retries against publishers, schedules | owns | — |
| Grouping deliveries into payloads, writing manifests | owns | validates |
| Temporary storage until acknowledged | owns | — |
| Permanent storage, dedup, identification, parsing | — | owns |
| Deciding whether a record is new, a duplicate or a new version | — | owns, and reports back |
| Licence terms per source | records `license_id` | stores it with every file |

Harvester never writes to Grimoire's database or storage directly, and Grimoire never calls publishers. On the NAS there is no Harvester: local ingest paths (`grimoire ingest`, Zotero import, a future upload form) build the same manifest internally and enter through the same code.

### Deployment roles and publisher adapters

Harvester is one codebase and one image that runs in one of two roles (D23). Only the inner Harvester speaks C1 to Grimoire.

| Role | Where it runs | What it does |
| --- | --- | --- |
| `all` | beside Grimoire, same compose stack | runs publisher API adapters, collects packages from edges, stages files, speaks C1 |
| `edge` | standalone on a DMZ host | receives SFTP drops or runs a listener; checks each package is complete; holds it until the inner Harvester has it |

**Edge to inner handoff (internal to Harvester):** the inner Harvester always fetches, by rsync over SSH from the edge. The edge only receives files from publishers and answers "what is ready?". `handoff.mode` sets only how the inner side learns that a package is ready.

| `handoff.mode` | How the inner side learns a package is ready | Connections the firewall must allow |
| --- | --- | --- |
| `poll` (default) | asks the edge every `handoff.poll_interval` | inner → edge (SSH, HTTPS) |
| `notify` | the edge sends a short "package ready" message; the inner side also polls on a long interval in case a message is lost | the same, plus one HTTPS endpoint edge → inner |

After fetching, the inner side checks the hashes against the manifest and then marks the package received on the edge, which deletes its copy. The edge never holds credentials for the inside network.

**Publisher adapters (D24):** each publisher is a small Python class over a shared base that handles auth headers, rate limits, paging, retries and staging. API keys and institutional tokens come from site config and never enter the repo. Elsevier comes first. ScienceDirect is Elsevier's full-text platform behind the same APIs, so the two are probably one adapter; check this against Elsevier's developer docs. Whether a config file alone can describe a publisher is decided once a second publisher exists.

### Contract C1 · Payload API

Harvester announces a payload; Grimoire pulls the files, verifies them, reports an outcome per record, and only a terminal status lets Harvester delete its copy. Base path: `/api/ingest/v1`.

| Method and path | Caller | Purpose | Success | Errors |
| --- | --- | --- | --- | --- |
| `POST /payloads` | Harvester | announce a payload with its manifest | `202` with `payload_id`, `status`, `status_url`; `200` if the same ID and identical manifest were already sent | `400` malformed JSON, `422` schema-invalid, `409` same ID with a different manifest, `413` too large, `401` |
| `GET /payloads/{payload_id}` | Harvester | status and per-record outcomes | `200` | `404` |
| `GET /payloads?status=&since=&cursor=` | Harvester | list payloads for reconciliation | `200`, paged | `400` |
| `POST /payloads/{payload_id}/retry` | Harvester or operator | re-fetch after Harvester fixed a file | `202` | `409` if already `stored` |

*Diagram: C1 payload status · 2 working states, 3 terminal (see the living doc).*

**Status response:**

```json
{
  "payload_id": "elsevier-tdm-2026-10-14-000123",
  "status": "partially_stored",
  "received_at": "2026-10-14T18:03:00Z",
  "finished_at": "2026-10-14T18:05:41Z",
  "records": [
    {"record_id": "r1", "outcome": "created", "item_id": "0192f1c2-…"},
    {"record_id": "r2", "outcome": "new_version", "item_id": "0192a7e0-…"},
    {"record_id": "r3", "outcome": "duplicate", "item_id": "0191ff03-…"},
    {"record_id": "r4", "outcome": "rejected", "error": {"code": "hash_mismatch", "message": "…"}}
  ]
}
```

Record outcomes: `created` (new item), `merged` (added to an existing item), `new_version` (new file for an existing item, e.g. a correction), `duplicate` (identical file already stored, nothing changed), `rejected` (with an error code).

**Optional callback:** if the manifest carries `callback_url`, Grimoire POSTs `{payload_id, status, at}` there on every status change. Polling stays authoritative; a missed callback must never lose data.

**Harvester's side of the contract:** every file `uri` answers `HEAD` and `GET` with `Range` support, and stays available until the payload is terminal.

### Manifest schema v1

A manifest is one payload: a source, and up to 1,000 records, each with identifiers, publisher metadata in CSL-JSON, and one or more files.

```json
{
  "manifest_version": 1,
  "payload_id": "elsevier-tdm-2026-10-14-000123",
  "created_at": "2026-10-14T18:02:11Z",
  "source": {
    "publisher": "Elsevier",
    "channel": "tdm-api",
    "license_id": "SITE-TDM-2025",
    "harvester_version": "0.4.2"
  },
  "callback_url": "https://harvester.example.org/api/callbacks/grimoire",
  "records": [
    {
      "record_id": "r1",
      "kind": "work",
      "identifiers": {"doi": "10.1016/j.anucene.2024.110123", "pii": "S0306454924001234", "issn": "0306-4549"},
      "metadata": {"type": "article-journal", "title": "…", "author": [{"family": "…", "given": "…"}], "issued": {"date-parts": [[2024, 3]]}, "container-title": "Annals of Nuclear Energy", "volume": "198", "page": "110123"},
      "files": [
        {"role": "fulltext_pdf", "uri": "https://harvester.example.org/files/…/a.pdf", "sha256": "…", "size_bytes": 1843211, "media_type": "application/pdf"},
        {"role": "fulltext_xml", "uri": "https://harvester.example.org/files/…/a.xml", "sha256": "…", "size_bytes": 402118, "media_type": "application/xml", "xml_profile": "elsevier-ja", "structured": true}
      ]
    }
  ]
}
```

| Field | Rule |
| --- | --- |
| `payload_id` | unique per Harvester, stable across retries; the idempotency key |
| `source.channel` | `tdm-api`, `sftp`, `manual`, `local` (NAS ingest paths) |
| `record.kind` | `work` (a paper, book, chapter, report), or a notice: `correction`, `retraction`, `withdrawal` that points at an existing work by identifiers |
| `identifiers` | at least one of `doi`, `pii`, `publisher_id`, `isbn`, `report_number`; DOI lowercased |
| `metadata` | CSL-JSON (the citation format Zotero and Crossref already use); trusted above GROBID extraction, below manual edits |
| `files[].role` | `fulltext_pdf`, `fulltext_xml`, `fulltext_epub`, `page_images`, `supplement`, `metadata_xml` |
| `files[].xml_profile` | `jats`, `elsevier-ja`, `springer-a++`, `other`; `structured: false` marks raw-body XML |
| Limits | ≤ 1,000 records and ≤ 5 MB of manifest JSON per payload; no limit on file sizes |

**Versions and corrections:** a record whose identifiers match an existing item but whose file hash is new becomes a new file version on that item (`new_version`); the old file stays stored and cited. A `retraction` notice flags the item, and retrieval results carry the flag.

### Transfer, integrity and errors

Grimoire trusts nothing it has not hashed itself, and every step can be repeated without changing the outcome.

- **Pull, not push.** Grimoire fetches each file from its `uri`; Harvester never uploads into Grimoire. Fetches resume with `Range` and run with bounded concurrency (default 4 per payload).
- **Integrity.** Grimoire checks `sha256` and `size_bytes` on every file. A mismatch rejects that record with `hash_mismatch`; the rest of the payload continues.
- **Idempotency.** Re-sending a `payload_id` with an identical manifest returns the existing status. A file whose hash is already stored is linked, not re-downloaded.
- **Atomicity per record.** A record is stored completely or not at all; a payload can end `partially_stored`.
- **Retries.** Grimoire retries transient fetch errors (timeouts, `5xx`) with backoff up to 5 times before rejecting the record with `fetch_failed`. `POST /retry` restarts failed records only.
- **Auth.** Harvester calls Grimoire with a bearer token issued per Harvester instance; Grimoire fetches files with a token Harvester issues, or short-lived signed URLs in the manifest.
- **Versioning.** The path carries the major version (`/api/ingest/v1`) and the manifest carries `manifest_version`. Within v1, changes are additive; Harvester must ignore unknown response fields, and Grimoire rejects unknown required fields with `422`.

**Error codes (stable):** `manifest_invalid`, `payload_conflict`, `payload_too_large`, `unauthorized`, `hash_mismatch`, `size_mismatch`, `fetch_failed`, `unsupported_media_type`, `identifier_missing`, `notice_target_unknown`.

### Conformance kit

The kit is what lets Harvester be developed elsewhere: it can be built and tested against the contract with no running Grimoire.

| Piece | Path in the Grimoire repo | Used by |
| --- | --- | --- |
| OpenAPI spec | `contracts/payload/v1/openapi.yaml` | both; generate clients and servers from it |
| Manifest JSON Schema | `contracts/payload/v1/manifest.schema.json` | Harvester validates before sending |
| Fixtures (\~12 payloads) | `contracts/payload/v1/fixtures/` | both: valid single and multi-record, XML + PDF, raw-body XML, correction, retraction, hash mismatch, duplicate, oversize, unknown field |
| Mock Grimoire | `grimoire contract mock payload --port 8099` | Harvester's CI: accepts payloads, fetches files, returns scripted outcomes |
| Mock Harvester file server | `grimoire contract mock files --dir fixtures/` | Grimoire's CI |
| Conformance suite | `pytest contracts/payload/v1/conformance/ --target <url>` | runs against either side's real implementation |

### Milestones and gates

- [ ] **H1 · Contract draft:** OpenAPI, JSON Schema and fixtures reviewed by Rodrigo; frozen as `v1.0-draft` for Harvester development to start.
- [ ] **H2 · Kit:** mock Grimoire, mock file server and conformance suite pass against each other in CI.
- [ ] **H3 · Grimoire implementation:** `/api/ingest/v1` on Postgres (needs R2's S1); all fixtures give their expected outcomes; re-sending every fixture changes nothing.
- [ ] **H4 · Local sources:** `grimoire ingest` and Zotero import go through the same manifest path; the v1 ingest tests still pass.
- [ ] **H5 · Real Harvester:** the Elsevier channel end to end at an institutional site; conformance suite green against the real Harvester; `v1.0` of the contract tagged.

### Open questions

- [ ] What does Harvester produce today, and how far is it from manifest v1?
- [ ] Does Harvester expose files over HTTPS, or does Grimoire need to read a shared directory for SFTP-delivered files?
- [ ] Which XML profiles do the first publishers deliver (JATS, Elsevier, Springer)?
- [ ] How long must Harvester keep `rejected` payloads for inspection before deleting?
- [ ] Does the site allow one HTTPS call from the DMZ host inward? If not, its edge stays on `poll`.
- [ ] Are Elsevier and ScienceDirect one adapter (same API, key and institutional token)?
- [ ] Which publisher is second, to test whether a config-only adapter is enough?

## R2 · Storage, RAG and cartography

R2 is Grimoire's core: it stores literature at every granularity with full provenance, builds the search indexes and the corpus map, and serves everything through contract C2, the Retrieval API.

### Boundary

| Direction | Contract | What crosses it |
| --- | --- | --- |
| Provides | C2 · Retrieval API | search results, items, full text at any granularity, provenance, map nodes, files |
| Consumes | C1 · Payload API | manifests and files from Harvester or local ingest paths |
| Consumes | C3 · LLM port | dedup judging, metadata fallback, `curated` search, map summaries |
| Consumes | C4 · Job contract | parsing, OCR, chunk embedding, map builds, run as batch jobs |

R2 owns Postgres and the file store; no other region writes to them. Query-time work that must be fast (embedding one query, reranking \~50 passages) runs inside Grimoire, not through runners.

### Contract C2 · Retrieval API

An HTTP JSON API is the canonical interface; the MCP server is a thin adapter that maps each tool onto one endpoint, so any future consumer (a chatbot, a web UI) uses the same contract Claude Code does. Base path: `/api/v2`. Read-only.

| Endpoint | Returns | MCP tool |
| --- | --- | --- |
| `POST /search` | ranked evidence units with provenance | `search` |
| `GET /items/{item_id}` | metadata, identifiers, relations, available files and artifacts | `get_item` |
| `GET /items/{item_id}/content?granularity=abstract\|section\|pages\|full&ref=` | text at the requested granularity, with page spans | `get_content` |
| `GET /items/{item_id}/structure` | section and chapter tree | `get_document_structure` |
| `GET /items/{item_id}/related?kind=` | preprint chain, editions, parts, semantic neighbours | `list_related` |
| `GET /items/{item_id}/citation?style=bibtex` | a formatted citation | `get_citation` |
| `GET /provenance/{unit_id}` | the full chain: unit → artifact → file → item, plus derivations | `provenance` |
| `GET /files/{sha256}` | the stored file (PDF, EPUB, XML); `#page=N` links work in viewers | (link only) |
| `GET /tags`, `GET /collections` | the library's tags and collection tree | `list_tags`, `list_collections` |
| `GET /map/locate?q=` | the map regions a question falls in | `map_locate` |
| `GET /map/nodes/{node_id}` | a region's summary, key papers, entities, neighbours, children | `map_node` |
| `GET /items/{item_id}/citations?direction=out\|in` | citing / cited items | `citations` |

Every one of v1's ten MCP tools keeps a home, so Claude Code on the NAS loses nothing: `get_snippets` becomes `search` with `unit: passage` and an item filter, `get_full_text` becomes `get_content`, and `find_by_tag` becomes `tag` and `collection` filters on `search`.

**Search request:**

```json
{
  "query": "resonance self-shielding in TRISO fuel",
  "mode": "filtered",
  "unit": "passage",
  "k": 20,
  "per_item_max": 3,
  "filters": {"year": {"gte": 1990}, "item_type": ["paper"], "venue": [], "map_node": [], "entity": ["TRISO"], "section_type": ["methods", "results"], "source_tier": ["institutional"]},
  "request_id": "…"
}
```

**Search response (one result shown):**

```json
{
  "request_id": "…",
  "mode": "filtered",
  "results": [{
    "unit": {"type": "passage", "id": "0192f2…"},
    "text": "…the Dancoff factor for a TRISO particle lattice…",
    "score": 0.83,
    "scores": {"bm25": 12.4, "dense": 0.71, "rerank": 0.83},
    "item": {"id": "0191ab…", "doi": "10.13182/NSE…", "title": "…", "authors": ["…"], "year": 2011, "venue": "Nuclear Science and Engineering", "flags": []},
    "provenance": {"chunk_id": "0192f2…", "section": "3.2 Dancoff factors", "pages": [7, 8], "artifact": {"id": "…", "tool": "publisher-xml", "tool_version": "1"}, "file": {"sha256": "…", "media_type": "application/pdf", "url": "/api/v2/files/…#page=7"}}
  }],
  "withheld": {"count": 14, "unit_ids": ["…"]},
  "usage": {"llm_calls": 0, "latency_ms": 412}
}
```

**Rules:** every result carries `provenance` down to a file and page range; `withheld` lists what filters or the `curated` LLM removed so a consumer can widen; `unit` may be `passage`, `item` or `map_node`; retracted items carry `flags: ["retracted"]`. The retrieval eval set (seeded from the v1 20-query oracle) is part of the contract's acceptance.

### Data model and provenance

Postgres (approved, D7) with pgvector and ParadeDB pg\_search holds everything; the v1 invariants that tests check today become constraints the database enforces. Each table is reviewed with Rodrigo as the S1 migration is written.

| Entity | Holds | Notes |
| --- | --- | --- |
| `items` | one bibliographic work | `source_tier` (personal / institutional), `identifiers` (DOI, PII, OpenAlex, ISBN), `flags` (retracted, corrected) |
| `files` | content hash, media type, size, storage URI, `license_id`, `received_via` (payload ID) | an item can have several files and several versions of one role |
| `payloads` | C1 deliveries | manifest, status, per-record outcomes |
| `item_artifacts` | derived outputs: TEI, Markdown, OCR text, MathML→LaTeX | unique on (input hash, tool, tool version): a re-parse is a new row, never an overwrite |
| `chunks` | retrievable passages | NOT NULL `artifact_id`, `page_start`, `page_end`; optional `bbox`, `section`, `section_type` |
| `item_embeddings`, `chunk_embeddings` | vectors | `vector(768)` / `halfvec(1024)`; model name + version per row |
| `item_relations` | dedup, editions, parts, citations | citations from publisher XML, OpenAlex, TEI |
| `entities`, `entity_mentions` | nuclides, codes, facilities, benchmarks, materials | dictionary-first extraction |
| `map_nodes`, `map_memberships` | cartography | per map build, with stable node IDs |
| `derivations` | links any derived unit (map summary, curated passage) to its source chunks and the `llm_call_id` that produced it | the provenance spine for LLM outputs |
| `llm_calls` (R3), `tasks` / `jobs` (R4) | logs and queues owned by those regions' code | same database, separate schema |

**Provenance chain (D25):** paragraph or chunk → section or chapter → file (sha256) → paper or book, plus the page range when the source has pages. Each chunk also records the artifact it came from (tool, version, input hash). LLM-derived units add → derivation → source chunks + LLM call.

**Section, not page, is the anchor:** publisher XML has section structure but no page breaks, and book chapters are what readers cite. So every chunk carries its section or chapter path, and text parsed from a PDF also carries its pages. No matching of XML text to PDF pages is needed.

**Metadata precedence:** manual edits, then publisher metadata from C1, then Crossref, arXiv, Open Library, GROBID, and the LLM fallback last. On a merge each field keeps its highest-ranked source, as in v1.

**Invariants** (v1 numbering kept):

| # | Invariant | Enforced by |
| --- | --- | --- |
| 1 | items + merge\_history = ingest records | trigger + test |
| 2 | no two live files share `sha256` | unique index |
| 3 | symmetric relations | deferred constraint trigger |
| 4 | embedding dimensions | column type |
| 5 | re-ingest is a no-op | unique keys on payload ID and file hash |
| 6 | dedup oracle precision and recall ≥ 0.95 | oracle test |
| 7 | papers with a DOI have a venue | check constraint |
| 8 | search index in step with rows | same-row index (structural) |
| 9 | `non_duplicate_pairs.a_id < b_id` | check constraint |
| 10 | no orphan chunks | foreign key, `ON DELETE CASCADE` |
| 11 | every chunk resolves to a file and page range | NOT NULL foreign keys + check |
| 12 | every derived unit lists at least one source chunk | trigger |
| 13 | an artifact is never overwritten | unique key on (input hash, tool, version) |

### Processing pipeline

Every step after storage is a versioned task submitted through R4, so the same pipeline runs on the NAS, an institutional HPC or AWS, and any step can be re-run with a new tool version without touching the others.

1. **Store** (in Grimoire, synchronous with C1): files hashed and stored, item created or merged using publisher metadata.
2. **Extract** (tasks): pick the best source per item, as in the table below.
3. **Chunk** (task): section-aware chunks from the chosen artifact, each with page span and section type.
4. **Embed** (task): SPECTER2 per item, BGE-M3 per chunk (model chosen on the eval set before the large backfill, D26). An item with no abstract, common among old scans, embeds its first ~500 words and is flagged for review.
5. **Enrich** (tasks): entity mentions, citation links, map placement.
6. **Index** (in Grimoire): collector loads outputs; search indexes update in the same transaction.

| Content | Body and equations from | Metadata and references from |
| --- | --- | --- |
| Structured publisher XML | XML; MathML converted to LaTeX | XML + OpenAlex / Crossref |
| Born-digital PDF, no XML | body extractor (Marker or Docling, by measurement) | GROBID |
| Scanned back issues | OCR only on pages without a text layer (ocrmypdf + Tesseract as the baseline; Mistral OCR or an open VLM, by measurement); page images and the searchable PDF kept as artifacts | OpenAlex / Crossref by DOI; GROBID fallback |
| Raw-body XML | weak text, used to check OCR output | as above |

GROBID flattens equations (`∫_{Ec}^{∞} σ_R(E) dE/E` came out as `∞ Ec σ R (E) dE E` in our TEI), so it is never the equation source.

**Test sets:** a parser set (\~50 papers: modern, equation-heavy, 1970s–80s scans) and an old-page set (100–200 hand-checked pre-XML pages), scored on text edit distance, normalized-LaTeX equation match and reading order.

**OCR baseline (measured 2026-04-21):** ocrmypdf + Tesseract read a 56-page 1982 NSE scan in 38 s (170 s CPU). Prose, title, authors and DOIs came out clean; Greek letters and math were mangled. That scan is the first page of the old-page set, and any OCR candidate must beat this baseline on equations.

**PDF → structured XML model (Exploration, own repository):** fine-tune an open vision-language model on recent page images paired with publisher XML, degraded to look like scans. Starts only if every off-the-shelf tool falls short on the old-page set, and only after the TDM agreements are checked for training rights.

### Search architecture

Search is two-stage — find candidate papers, then search only their passages — which keeps one Postgres instance fast at \~1M papers without an approximate index over \~50M chunks.

1. **Paper stage:** BM25 over title, abstract and full text (pg\_search) plus SPECTER2 paper vectors (pgvector HNSW), fused with RRF (k = 60, as in v1); filters apply here. Output: \~200 candidate papers.
2. **Passage stage:** BGE-M3 chunk vectors and BM25 restricted to the candidates' chunks, compared exhaustively; RRF again.
3. **Rerank** (`filtered`, `curated`): cross-encoder over the top \~50 passages, then at most `per_item_max` per paper.
4. **LLM curation** (`curated` only, via C3): query rewrite or split, map routing, relevance judgment, trimming passages to the relevant sentences.

| Mode | Steps | Latency target (institutional server, first guess) |
| --- | --- | --- |
| `raw` | 1–2 | < 200 ms |
| `filtered` | 1–3 | < 1.5 s |
| `curated` | 1–4 | < 15 s |

`curated` returns evidence, never prose answers, and reports what it withheld. Each mode has its own recall oracle; `curated` must beat `filtered` by a stated margin to justify its cost, and degrades to `filtered` (with a flag) when the LLM budget is exhausted.

### Cartography layer (Exploration)

Hypothesis: a hierarchical map of the corpus, built mostly from the graph scholarly literature already has, gives GraphRAG's big-picture answers and query routing at a small fraction of GraphRAG's indexing cost.

| Approach | Graph built from | LLM cost at indexing | Fit here |
| --- | --- | --- | --- |
| Microsoft GraphRAG | LLM entity and relation extraction on every chunk, then community summaries | one call per chunk (tens of billions of tokens at \~50M chunks) | too costly at institutional scale; possible on the personal library |
| LazyGraphRAG | noun-phrase concept graph; LLM work deferred to query time | about the same as vector RAG | worth a pilot |
| HippoRAG 2 | knowledge graph + personalized PageRank at query time | per-chunk extraction | pilot on one journal only |
| RAPTOR | recursive clustering + summaries of chunks | one call per cluster | ideas reused in the map |
| **Cartography (proposed)** | citations + SPECTER2 similarity + dictionary entities, clustered into a hierarchy; LLM summaries per region only | thousands of calls | primary candidate |

**Structure:** Field → Subfield → Topic region → Paper → Section → Passage. Each map node holds an LLM-written summary, defining papers, key entities, time span and neighbouring regions; summaries record their sources in `derivations`.

**Inputs:** citations (publisher XML, OpenAlex, TEI; DOI-less references need fuzzy matching), paper embeddings, and dictionary-matched entities (nuclides, reactors and facilities, codes such as MCNP, Serpent, OpenMC, MOOSE, RELAP, benchmarks such as ICSBEP and IRPhE, materials), with an LLM only for ambiguous mentions.

**Citation build:** DOIs are normalized (resolver prefix stripped, lowercased), self-citations are skipped, and a re-run adds nothing. Each build reports edges added, references whose DOI is not in the corpus, and references with no DOI; the last two counts size the fuzzy matching (#19) needs.

**Uses:** routing for `curated`; big-picture questions answered from region summaries plus key papers; `map_locate` and `map_node` in C2 for agents to browse.

**Maintenance:** new papers join their nearest region at ingest; the full map is rebuilt periodically as an R4 job (graph clustering in igraph or graph-tool). Node IDs carry over between builds by membership overlap.

**Exit criteria:** on a pilot journal, map routing improves `curated` recall@10 over `filtered`, and domain readers judge region summaries accurate on a sample of 30. Rodrigo's earlier cartography notes get folded in when found.

### Migrating the v1 library

The 15,547-item personal library moves from SQLite to Postgres once; derived data that was never built (chunks, embeddings, TEI) is produced afterwards through R4 rather than migrated.

- Carry over: items, identifiers, relations, merge history, collections, `non_duplicate_pairs`, file records (the 42 GB of files stay where they are; only their records move).
- Rebuild, don't migrate: search indexes, chunks, embeddings, artifacts (only 2 TEI exist today).
- Verify: row counts and a field-by-field comparison on a 500-item sample; the v1 oracles (metadata ≥ 97%, dedup P/R ≥ 0.95, hybrid recall@10 ≥ 90%) must hold after the rebuild.
- The SQLite database is kept read-only until the comparison passes.

### Milestones and gates

- [ ] **S1 · Postgres core:** schema reviewed table by table; invariants 1–13 enforced; v1 library migrated and verified.
- [ ] **S2 · Retrieval API v2:** C2 OpenAPI spec frozen as draft; `raw` and `filtered` modes; MCP adapter; Claude Code can run the v1 research walkthrough through it; recall and latency measured.
- [ ] **S3 · Pipeline as tasks:** extract, chunk and embed run as C4 jobs on the `local` runner; the real-library backfill completes; recall oracle re-run.
- [ ] **S4 · Better sources:** publisher-XML path, body extractor and OCR chosen by the parser and old-page test sets.
- [ ] **S5 · Institutional pilot journal:** one full journal archive through C1 → S3 on the HPC runner; publisher-XML coverage measured per year.
- [ ] **S6 · `curated` mode:** needs R3's L4; beats `filtered` on the eval set by the agreed margin.
- [ ] **S7 · Cartography:** exploration on the pilot journal; exit criteria met or closed with findings.
- [ ] **S8 · Full corpus:** all oracles hold at scale.

### Open questions

- [ ] Can each target site's Postgres run pgvector and ParadeDB pg\_search?
- [ ] Keep BGE-M3 for chunks, or switch (e.g. Qwen3-Embedding) before the large backfill? Decide on the S3 eval.
- [ ] Which journal is the pilot (e.g. the full NSE archive)?
- [ ] Should the retrieval eval set get a paraphrased-question set (no word overlap) to test semantic recall properly?
- [ ] Where are the earlier cartography-RAG notes?

## R3 · LLM inquiry

R3 gives Grimoire one way to ask an LLM a question, whether the model is reached directly (Anthropic API, Bedrock) or through a gateway (LiteLLM, internal models); contract C3 is the in-process port every Grimoire feature calls.

### Boundary

- **In scope:** text and structured-output generation for Grimoire's own tasks: dedup judging, metadata fallback, query rewriting, relevance judgment, passage trimming, map summaries, entity disambiguation, and VLM page OCR when an OCR tool is an LLM.
- **Out of scope:** embeddings and rerankers (local models, owned by R2 and run in-process or through R4); answer generation for users (the query layer's job).
- **Bulk work** (thousands of map summaries) still goes through C3, using a provider's batch API when the adapter supports it; R4 is not involved unless a local model is run on the HPC.

### Contract C3 · LLM port

Two layers: Grimoire features call named **tasks**; tasks call a provider-neutral **completion port**; adapters implement the port. Swapping provider or gateway is configuration, never a code change in a feature.

```python
# Task layer: what features call
result = llm.run("dedup_judge", DedupJudgeInput(a=item_a, b=item_b))
# -> TaskResult[DedupJudgeOutput] with .output, .call_id, .usage, .model

# Port: what tasks call; what adapters implement
class CompletionPort(Protocol):
    capabilities: Capabilities          # structured_output, prompt_caching, batch, vision
    def complete(self, req: CompletionRequest) -> CompletionResponse: ...
    def submit_batch(self, reqs: list[CompletionRequest]) -> BatchHandle: ...  # if capabilities.batch
    def batch_results(self, handle: BatchHandle) -> list[CompletionResponse]: ...
```

| Object | Fields |
| --- | --- |
| Task definition | `name`, `version`, prompt template, input and output schemas (Pydantic → JSON Schema), `model_role`, eval set path |
| `CompletionRequest` | `model_role`, `system`, `messages` (text and images), `output_schema` (optional), `max_tokens`, `temperature`, `cache_hints`, `request_id`, `task` + `task_version` |
| `CompletionResponse` | `content`, `parsed` (validated against the schema), `usage` (input, output, cache read, cache write tokens), `model_id`, `stop_reason`, `latency_ms`, `provider_request_id` |
| Model roles (config) | e.g. `judge`, `router`, `summarizer`, `ocr` → adapter + model ID per deployment |

**Rules:** structured outputs are validated against the task's schema, with one repair retry before failing; every call has a `request_id` for idempotent retries; features never see provider-specific types.

### Adapters

Two real adapters cover every target named so far, plus a fake for tests; Grimoire does not depend on the LiteLLM library itself (this supersedes D10).

| Adapter | Speaks | Targets | Structured output | Prompt caching | Batch |
| --- | --- | --- | --- | --- | --- |
| `anthropic` | Anthropic Messages API (official SDK) | Anthropic API (NAS); Claude on Bedrock via the SDK's Bedrock client; any gateway exposing the Messages API | yes | yes | yes on the Anthropic API; Bedrock batch to be checked |
| `openai_compatible` | OpenAI Chat Completions | a LiteLLM gateway; vLLM or similar serving internal models | when the backend supports JSON-schema output | backend-dependent, assume no | no |
| `fake` | recorded responses | tests and CI | yes | — | yes |

**Example configuration (institutional site):**

```yaml
llm:
  adapters:
    gateway: {type: openai_compatible, base_url: https://litellm.example.org/v1, api_key_env: LITELLM_KEY}
    internal: {type: openai_compatible, base_url: https://models.example.org/v1}
  roles:
    judge:      {adapter: gateway, model: claude-haiku}
    router:     {adapter: gateway, model: claude-haiku}
    summarizer: {adapter: gateway, model: claude-sonnet}
    ocr:        {adapter: internal, model: <vlm-name>}
```

On the NAS the same roles point at one `anthropic` adapter with an API key. Model names above are placeholders for whatever the gateway exposes.

### Call log, provenance and limits

Every LLM call is recorded, and anything an LLM writes into Grimoire's data points back to the call that produced it.

- **`llm_calls` table:** call ID, task + version, prompt hash, adapter, model ID, usage, estimated cost, latency, outcome, timestamp. The full prompt and response are kept when the output is persisted (a map summary, a dedup decision), and for a configurable retention period otherwise.
- **Provenance:** a persisted LLM output links through `derivations` (R2) to its source chunks and its `llm_call_id`, so a summary can be traced to both its evidence and the model that wrote it.
- **Budgets:** per-role token budgets per day; when a budget runs out, `curated` search falls back to `filtered` with a flag, and batch tasks pause rather than fail.
- **Retries and timeouts:** backoff on `429` and `5xx`, at most 3 attempts, a timeout per role; the `request_id` makes retries safe.
- **Caching:** prompt caching on the `anthropic` adapter for long shared prefixes (task instructions, region context); an optional response cache keyed by prompt hash for deterministic tasks such as dedup judging.
- **Batch:** map summaries and other bulk tasks use `submit_batch` when available, at roughly half the per-token price on the Anthropic API.

### Task catalogue

Each task ships with a versioned prompt, a schema and an eval set; a prompt change that lowers the eval score does not merge.

| Task | Used by | Output | Eval set | Exists in v1 |
| --- | --- | --- | --- | --- |
| `dedup_judge` | R2 dedup (ambiguous band 0.90–0.97) | same work? + reason | the 100-pair dedup oracle | yes, `dedup_llm.py` |
| `metadata_fallback` | R2 identification when Crossref, arXiv and Open Library fail | CSL-JSON fields | Phase 1 metadata oracle misses | yes, `resolve/llm_fallback.py` |
| `query_plan` | `curated` search | rewritten query, sub-queries, extracted filters | retrieval eval set | no |
| `relevance_judge` | `curated` search | keep / drop + score per passage | retrieval eval set with graded labels | no |
| `passage_trim` | `curated` search | sentence spans to keep | hand-labelled sample | no |
| `region_summary` | cartography | summary + key papers + entities | reader panel, 30 regions | no |
| `entity_disambiguate` | entity extraction | entity ID or none | hand-labelled mentions | no |
| `page_ocr` | R2 extraction, if a VLM is chosen | Markdown + LaTeX per page | old-page set | no |

### Milestones and gates

- [ ] **L1 · Port:** task and port interfaces, `fake` adapter, `llm_calls` log; contract frozen as draft so R2 can code against it.
- [ ] **L2 · Anthropic adapter:** direct API on the NAS; structured output and caching verified.
- [ ] **L3 · Existing tasks moved:** `dedup_judge` and `metadata_fallback` run through the port; dedup oracle unchanged (P/R ≥ 0.95).
- [ ] **L4 · Gateway adapter:** `openai_compatible` against a LiteLLM gateway; the same task evals pass on both adapters.
- [ ] **L5 · Query tasks:** `query_plan`, `relevance_judge`, `passage_trim` with evals; unblocks R2's S6.
- [ ] **L6 · Batch:** `submit_batch` on the Anthropic adapter; `region_summary` at scale for R2's S7.

### Open questions

The gateway questions are answered by probing the gateway from inside the deploying site while building L4 (D27).

- [ ] Which models, rate limits and cost accounting does the site's gateway expose?
- [ ] Does the gateway also expose the Anthropic Messages format (keeps prompt caching and batch), or only OpenAI format?
- [ ] Is batch inference available through the gateway or Bedrock for bulk map summaries?
- [ ] Which internal models are available, and is any of them a vision model usable for OCR?

## R4 · Runners

R4 moves heavy work off the Grimoire server: Grimoire queues tasks, batches them, and pushes each batch to a configured runner, which starts a container that knows nothing about Grimoire beyond contract C4.

### Boundary

| Role | Owns | Knows about |
| --- | --- | --- |
| Task author (R2, e.g. the GROBID or embedding step) | a job image and its task definition | C4 only |
| Dispatcher (Grimoire) | queue, batching, choosing a runner, tracking status | tasks, runner adapters |
| Runner adapter | submitting, polling, cancelling on one platform | C4 + its platform (Docker, Slurm, AWS Batch) |
| Job container | doing the work, writing outputs and a result file | C4 only; no database, no Grimoire code required |
| Collector (Grimoire) | validating results and loading them into Postgres | C4 + R2's data model |

Nothing has to stay online between jobs: runners are asked to start work, and the collector reads what finished.

### Contract C4 · Job contract

A job is one container run over one batch: it reads a JobSpec, processes each task in it, writes outputs to the given location, and ends with a result file listing what happened to every task.

**JobSpec v1** (passed as `/job/spec.json`, or by URI in `GRIMOIRE_JOB_SPEC`):

```json
{
  "spec_version": 1,
  "job_id": "0192f5…",
  "task": {"name": "grobid-fulltext", "version": "0.8.1-1"},
  "image": "registry.example.org/grimoire/grobid-fulltext@sha256:…",
  "params": {"consolidate_header": false},
  "tasks": [
    {"task_id": "0192f5…a", "input_hash": "…", "inputs": [{"name": "pdf", "uri": "https://grimoire.example.org/api/jobs/v1/blobs/…?sig=…", "sha256": "…"}]}
  ],
  "output": {"uri_prefix": "https://grimoire.example.org/api/jobs/v1/outputs/0192f5…/", "mode": "signed_put"},
  "resources": {"cpus": 8, "mem_gb": 16, "gpus": 0, "walltime_min": 120}
}
```

**Result manifest** (written last, at `<uri_prefix>/result.json`):

```json
{
  "job_id": "0192f5…",
  "tasks": [
    {"task_id": "0192f5…a", "status": "ok", "outputs": [{"name": "tei", "uri": "…/0192f5…a/tei.xml", "sha256": "…", "media_type": "application/tei+xml"}], "metrics": {"seconds": 2.4}},
    {"task_id": "0192f5…b", "status": "failed", "error": {"code": "grobid_timeout", "message": "…", "retryable": true}}
  ],
  "runtime": {"image_digest": "sha256:…", "host": "…", "started_at": "…", "finished_at": "…"}
}
```

**Container rules:**

- One OCI image per task family, runnable unchanged by Docker and Apptainer; no root, no writes outside `/tmp` and `/job`.
- Per-task failures go in the result file; the process exits `0`. A non-zero exit means the whole job failed and may be retried.
- Outputs are keyed by `task_id`, so re-running a job overwrites the same objects; `result.json` is written last and marks completion.
- Network access only to the input and output URIs, and to a declared service such as a GROBID sidecar inside the job.
- The image declares its task name, version and default resources in labels, so the dispatcher needs no separate registry.

### Runner interface and task lifecycle

A runner adapter is three calls plus a log fetch; everything platform-specific lives behind them.

```python
class Runner(Protocol):
    name: str                                   # "local", "slurm", "aws-batch"
    capabilities: RunnerCapabilities            # gpus, max_walltime, image_format, staging_modes
    def submit(self, spec: JobSpec) -> RunHandle: ...
    def status(self, handle: RunHandle) -> RunStatus: ...   # pending | running | exited(code) | lost
    def cancel(self, handle: RunHandle) -> None: ...
    def logs(self, handle: RunHandle, tail: int = 200) -> str: ...
```

The dispatcher picks a runner by deployment config and task needs (GPU, walltime), forms batches of 500–5,000 tasks, submits, and polls `status`. The collector then reads `result.json`.

*Diagram: task lifecycle · 7 states, retry loop (see the living doc).*

A batch whose job exits non-zero or is `lost` sends all its unfinished tasks to `failed`; a task marked `failed` in `result.json` goes there alone. Tasks are re-batched on retry, so one bad input cannot hold back the others.

### Data staging and failure handling

Jobs reach their data through URIs, so no runner needs shared storage with Grimoire; the staging mode is chosen per runner.

| Staging mode | How inputs and outputs move | Good for |
| --- | --- | --- |
| `signed_get` / `signed_put` (default) | short-lived signed HTTPS URLs to Grimoire's job blob endpoints; the job needs no credentials | NAS, an HPC whose compute nodes can reach the server, AWS |
| `s3` | an S3-compatible bucket both sides can reach; Grimoire uploads inputs before submitting | AWS Batch; an HPC site with a shared object store |
| `posix` | a staging directory copied in and out by a transfer step (rsync or Globus) | an HPC whose compute nodes cannot reach the server |

- **Idempotency:** a task is unique on (task name, version, input hash); a completed one is never queued again. Outputs are keyed by `task_id`, so a re-run overwrites in place.
- **Partial results:** the collector loads every `ok` task from a `result.json` even if others failed.
- **Lost jobs:** a job not seen in `status` for longer than its walltime plus a grace period is marked `lost`; its tasks go to `failed`.
- **Poison inputs:** a task that fails three times goes to `dead_letter` and is listed in an operator report; it never blocks its batch again.

### Adapters

| Adapter | Submits with | Status from | Image format | Scales to zero | Where |
| --- | --- | --- | --- | --- | --- |
| `local` | `docker run` (or Podman), bounded concurrency | container exit code | OCI | yes | NAS, laptop, institutional server |
| `slurm` | `sbatch` job array via slurmrestd, or SSH to a login node | `sacct` / slurmrestd | Apptainer `.sif` built from the OCI image | yes | institutional HPC |
| `aws-batch` | `SubmitJob` on a spot compute environment | `DescribeJobs` | OCI in ECR | yes | AWS, for one-off backfills or overflow |

GPU tasks (chunk embedding, VLM OCR) declare `gpus` in resources; adapters map it to `--gpus`, Slurm `--gres` or a GPU compute environment.

**Sizing:** GROBID full text was estimated at 3–10 s per paper on CPU. That is 12–42 h for the 15k library on the NAS `local` runner, but roughly 35–115 CPU-days for 1M papers, so an institutional backfill needs `slurm` or `aws-batch`.

### Conformance kit

| Piece | Path | Used by |
| --- | --- | --- |
| JobSpec and result JSON Schemas | `contracts/jobs/v1/` | task authors, runner authors |
| `grimoire job validate <image>` | CLI | task authors: runs the image on fixture inputs with the `local` runner and checks `result.json` against the schema, re-run idempotency and the exit-code rule |
| `grimoire runner check <name>` | CLI | runner authors: submits a no-op image, checks status transitions, cancel and logs |
| Reference image `grimoire/job-echo` | registry | hashes its inputs and writes them back; the smallest valid job |

### Milestones and gates

- [ ] **J1 · Contract draft:** JobSpec and result schemas, staging modes, reference image; frozen as draft so task authors can start.
- [ ] **J2 · Local runner:** dispatcher, collector and `local` adapter; `job-echo` passes `job validate`; a killed job retries and ends `collected`.
- [ ] **J3 · First real images:** GROBID full text and chunk embedding packaged per C4 (task code from R2); unblocks R2's S3.
- [ ] **J4 · Slurm runner:** on an institutional HPC; the same batch gives byte-identical outputs on `local` and `slurm`.
- [ ] **J5 · AWS Batch runner:** a backfill batch on spot instances; the compute environment is back at zero afterwards.

### Open questions

These are answered by probes at the deploying site; the Slurm adapter is built there, and must pass the same runner conformance check as the local adapter (D27).

- [ ] Can the HPC's compute nodes reach the server, or only login / data-transfer nodes? This picks the staging mode.
- [ ] Is slurmrestd available, or must the adapter submit over SSH with a service account?
- [ ] Is there an S3-compatible object store or Globus endpoint reachable from both the VM and the HPC?
- [ ] Where do job images live at the site (an internal registry), and can the HPC pull from it?
