---
title: "docs_record's docs/README.md and unlisted notes break mkdocs build --strict"
kind: "gotcha"
summary: "`docs_record` writes `docs/README.md` alongside the note it was asked for."
tags: [docs, mkdocs, gates, docs_record]
created: "2026-11-22T13:47:01.809Z"
updated: "2026-11-22T13:47:01.809Z"
---
`docs_record` writes `docs/README.md` alongside the note it was asked for. `mkdocs build --strict` — one of this project's gates — then fails, because `docs/README.md` "conflicts with `index.md`":

```text
WARNING - Excluding 'README.md' from the site because it conflicts with 'index.md'.
Aborted with 1 warnings in strict mode!
```

A second failure follows it: a new note that is not in `mkdocs.yml`'s `nav` list is reported as "exists in the docs directory, but are not included in the nav configuration", which strict mode also treats as fatal.

## What to do

1. Delete `docs/README.md` (`rm docs/README.md`) — it is not wanted in the repository; the index is built from the front matter.
2. Add one line per new note to `mkdocs.yml`'s `nav`, e.g.

```yaml
  - Project knowledge:
      - Tutorials ROOT vetoes for no display: instructions/running-the-tutorials-root-vetoes-for-no-display-also-vetoed.md
      - /tmp is wiped mid-session: gotchas/tmp-is-wiped-mid-session-keep-root-s-sources-and-every-run.md
```

3. Then `mkdocs build --strict` is clean, and commit the note together with the `mkdocs.yml` line. `exclude_docs: history/` in `mkdocs.yml` is what keeps the superseded plans out of both the site and the nav warning.
