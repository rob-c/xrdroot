# docs

What this project has learned, written down so nobody has to rediscover it.

Each file here answers one question a new contributor (or a new agent on a new
machine) would otherwise answer wrongly:

- `rules/` — hard constraints. Do not break these.
- `instructions/` — how work should be done in this project.
- `gotchas/` — things that break in non-obvious ways.
- `discoveries/` — how something actually behaves, where that is surprising.
- `decisions/` — a choice that was made and the reason.
- `architecture/` — how the pieces fit together.

Every file starts with YAML front matter (`title`, `kind`, `summary`, `tags`,
`updated`) so it can be indexed cheaply. Read the front matter first; it is
usually enough.

**Only files with a front-matter `kind:` are indexed.** A file without one is
reported as "not indexed" rather than silently ignored, because this folder is
sometimes already a documentation site or an ADR collection that this extension
does not own. To add a hand-written page to the index, give it the front matter
above; `docs_record` writes it for you.

This file is maintained by the `pi-docs-first` extension, but it is ordinary
Markdown — edit it by hand whenever it is wrong.
