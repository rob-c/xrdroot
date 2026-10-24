# Checking component boundaries

The prepared tool is `tools/components.py`. Its policy lives under
`[tool.xrdroot.components]` in the existing `pyproject.toml`. These commands
operate on the selected single source tree; they do not build component wheels
or install separate packages. Run them only in the appropriate scheduled
validation slot. The source-preparation status in the [overview](index.md)
applies to these commands as well.

## Ownership and imports

Every source module has one owner. Exact module assignments take precedence over
package-prefix assignments, allowing compatibility paths such as `geom.paint`
to be owned by graphics. The map also assigns the app facade. Core owns the
repository's `py.typed` marker; the import inventory covers Python source
modules. Direct dependencies form a DAG; aggregate `@engines` and `@facade`
entries expand to the explicit component sets.

The checker visits imports inside functions as well as top-level imports.
Making a forbidden import lazy does not make its component edge legal. External
hard and optional dependencies are declared per owner; optional imports remain
lazy, and private third-party imports are reported. Dynamic import sites need
narrow source-reviewed declarations. [Provider targets](providers.md) are checked
against a separate extension-edge allowlist.

## Inventory, ratchet and graph

From the worktree root, with its environment active:

```bash
export PYTHONPATH="$PWD/src"
python tools/components.py graph
python tools/components.py graph --dot
python tools/components.py check --inventory /absolute/durable/path/component-inventory.json
```

`--inventory` creates a new unreviewed report and refuses to overwrite a file.
Its successful exit means the inventory was written, even if it reports debt.
It does not approve that debt or pass a boundary gate. The report records actual
source hashes and the policy fingerprint.

After source review of the exact inventory and policy:

```bash
python tools/components.py check
python tools/components.py check --strict
```

Normal `check` requires an explicitly reviewed baseline with a review record and
matching policy. It rejects new violations and stale allowances for resolved
ones; resolved debt must shrink in the reviewed inventory. Moving a line alone
does not forgive a violation. `--strict` requires zero violations independently
of baseline allowances. The inherited prepared inventory is unreviewed, and the
latest tooling-source revision has no accepted actual inventory yet.

Use the global root option before the command to inspect another worktree:

```bash
python tools/components.py --root /absolute/path/to/worktree graph
```

Do not replace a failed inventory or simply mark a report reviewed to make a
command pass. Ownership and provider-policy changes require a fresh attributed
inventory and review.

## Fresh-process isolation

```bash
python tools/components.py isolate cppstd core
python tools/components.py isolate cppstd --tests cppstd
```

Each named component runs in its own fresh child process. The blocker permits
its dependency closure and the inert parent `xrdroot` bootstrap, then imports
its owned modules from the selected source path. It rejects cached package
escapes and a module loaded from another source tree. A declared compatibility
module alias must return the same actual module object at the exact reviewed
source origin; an arbitrary module with similar attributes is not an exception.
External dependencies are checked against the allowed closure; a third-party library's own internal
imports do not grant private imports to xrdroot code.

`--tests cppstd` selects the declared P2 test scope named `cppstd`, rather
than an arbitrary pytest path. Its policy declares `components = ["cppstd"]`,
`paths = ["tests/cppstd"]`, `support = ["tests/cppstd/conftest.py"]`,
`confcutdir = "tests/cppstd"` and `blocked = ["xrdclient"]`. The explicit
fixture cutoff keeps broad root fixtures from silently loading higher engines.
This interface requires the prepared P2 scope successor in the integrated tool.

Choose scopes whose fixtures and helpers stay within the closure. A mixed
engine/ROOT-adapter test belongs in integration coverage and will legitimately
fail a lower-only boundary. Do not remove assertions or relabel such a test as
isolated coverage. The facade's inert bootstrap is a prerequisite; unintegrated
source packets cannot claim isolation by themselves.

## Acceptance beyond the boundary tool

The component commands supplement the existing normal and forced-OTHER tests,
strict 100% line/branch coverage, Ruff, strict mypy, maintainability and strict
docs. They do not replace numerical fixtures, image comparisons, affected
tutorials, the translator's unchanged 98% floor with no crashes, or the global
no-PASS-lost tutorial ratchet. Limits and assertions remain unchanged.

Keep source maps, reviewed inventories, reports and failed attempts under a
unique durable run/recovery directory. A source-only packet or documentation
example is not execution evidence. Separate-project packaging remains deferred.
