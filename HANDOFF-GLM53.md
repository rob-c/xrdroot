# Development handoff to GLM-5.3

Resume as coordinator of the existing work. Read the current user instructions,
AGENTS.md, MODULARISATION.md and this supplement before making changes. The active
objective is internal modularisation in one Python project and distribution,
including `xrdroot.cppstd`. Separate distributions are deferred. Tutorial parity
and its publication restrictions remain in force.

## Recover all lanes

The durable evidence root is `/Users/rcurrie/xrdroot-parity` (called B below);
R means `B/recovery`. Read `B/RECOVERY-STATE.json` and follow its
`latest_coordinator_checkpoint` reference, verifying its SHA. Read `B/PLAN.md`,
`B/STATUS.md`, and the checkpoint's referenced lane records to recover the
modularisation, tutorial parity, reader/translator, performance, persistence,
harness and second-oracle work. Older top-level active fields and historical
paragraphs may be stale. This supplement records progress after the checkpoint;
inspect actual saved completions before launching anything.

Keep important evidence under B or this repository, never disposable `/tmp`.
Preserve every original failed attempt and its exact source. Do not pool
different sources or replace a failed capture with a diagnostic PASS.

## Git and acceptance

- Repository: `/Users/rcurrie/src/xrdroot`. Local main remains
  `e692236d8ff54916649cdcb613d36908bd30ba82`; published main remains
  `fa34ea9d4871b44cbbede35bf4f726b95c3c52ec`.
- Implementation worktree:
  `/Users/rcurrie/src/xrdroot-wt/modularisation-gated-commits-20261021`.
  Branch `refactor/modularisation-commits-20261021` is clean at
  `4fe45f164e590e2c1962ee80fe0bb7f304560da1`: six local implementation commits.
  Their record is `R/modularisation-03-L3-local-commit-ROOT-ACTUAL-20261021/committed.json`.
- Fifteen ordered prefixes have independently accepted own-source six gates.
  The separately gated complete candidate does not replace any prefix's gates.
  All 25 prepared source worktrees are listed in
  `R/modularisation-selected25-reviewed-baselines-materialized-ROOT-ACTUAL-20261021/ordered-summary.json`.
- The committed tutorial baseline is unchanged. No main publication until all
  required gates, the global no-PASS-lost ratchet and CI pass. Existing losses,
  including rf102/rf404/rf406 Python, are not waived. The user rejected a waiver;
  the original October 15 captures were lost in an OS crash with no archive.
- AGENTS.md and MODULARISATION.md were already untracked local instruction/plan
  files. Do not accidentally stage them or unrelated work into a fix commit.

## Newly completed step 16: audit next

`B/runs/modularisation-ordered-prefix-16-istreams-all6-FIRST-20261021/complete.json`
now records exit 0, all ten stages exit 0, unchanged source, drained owned
processes and slot released. It finished `2026-10-22T00:00:38.767028Z` after the
last saved coordinator checkpoint. Both modes record 9,680 passed, one skip and
46 expected failures, with strict 100% coverage over 92,541 statements and
22,596 branches. This completion has NOT yet received its saved six-gate audit.

Its prepared audit chain is
`R/modularisation-16-istreams-saved-audit-inputs-PREPARED-SOURCE-20261022/`:

- `INPUTS.json` SHA `3f701856b6093cd31f330f922db0935d183dc46f8b18ed44529cc7dba39cc128`.
- `execute-after-completion.py` SHA
  `1418e278fefbf6f35ce776b0e6cdc1aae7db2b2cffdc8c3c63a4857d9ebd3370`.
- Independent source review:
  `R/modularisation-16-istreams-audit-input-chain-independent-SOURCE-REVIEW-v2-20261022/review.json`,
  SHA `d3a1e73ab8efe7a63ed2d772fe59374c2cc91ecd68fa6f5a52d20b51c4424dcd`.

Verify the review and own completion, then explicitly authorize this existing
guarded saved-data audit. Do not rerun gates merely because the session changed.
Finalize step 16's own focused-basis record only after its actual audit passes.

## Math controls: preserve failure, use fresh preflight

The four original quiet comparisons remain FAIL; audit `3f172a86` accepts capture
integrity only. The controls655 FIRST run stopped before any tutorial stage:
its native-reader guard observed the other gate job's simultaneous `lsof`.
It is drained and released, with the failed startup intact at
`B/runs/modularisation-04-L4-affected659-controls655-FIRST-20261021/`.
Its saved failure audit is
`R/modularisation-04-controls655-FIRST-startup-failure-saved-audit-ACTUAL-20261022/receipt.json`,
SHA `1283e963038a149a1a17e9f2c9000e1775635d119446cb885dbd8af25a133b08`.
No tutorial ran in that attempt.

The old preflight receipt binds FIRST's proof output path. It cannot be reused
with a changed manifest, and the preserved FIRST directory must not receive
new writes. ROOT authorized a fresh output-only manifest and genuine cached
preflight using the unchanged program, inputs, native cache and comparison
settings. No oracle refresh or native science rerun is authorized by this step.

Prepared chain:
`R/modularisation-04-controls655-SECOND-fresh-preflight-chain-PENDING-SOURCE-20261022/`.
It contains an outer HELD three-stage preflight and HELD controlsSECOND template;
the future receipt hash is still zero. Independent chain review:
`R/modularisation-04-SECOND-fresh-preflight-chain-independent-SOURCE-REVIEW-20261022/review.json`,
SHA `db5e27b79fb5cbfb3e487fd04cad118bdd509ea5d4b869673a879010979b2e53`.

The metadata-only deriver has now been authority-bound but NOT executed:
`R/modularisation-04-SECOND-preflight-deriver-AUTHORITY-BOUND-SOURCE-20261022/derive-preflight.py`,
SHA `040e3a437e716897ebfde7a7def01b283ef0ecb11964f873dbd6964c693aa350`.
Its independent review passed 55 checks:
`R/modularisation-04-SECOND-bound-preflight-deriver-independent-SOURCE-REVIEW-20261022/review.json`,
SHA `7d3e1155498124a9a58287b2bd14624e10bcecde81b89763bcdb153f6d0cfb62`.
Next: verify and execute only this metadata derivation, prepare/review a saved
fresh-preflight auditor, and recover fresh resource authority before dispatch.

The plan pairs fresh preflight slot 0 with original step17 gates slot 1 AFTER
step16 completion/audit. Start preflight first; wait for its preflight and
after-acquire native checks before starting step17, avoiding synchronized
startup observers. Any later active-peer release needs its own reviewed proof;
otherwise wait for both slots to be free. No guard relaxation is permitted.

The saved audit/join/review family still needs a reviewed split-lineage
successor: original quiet evidence keeps old manifest `debe3f61`, receipt
`d041841a`, auditor `36be5ec4` and audit `3f172a86`; future SECOND controls use
their own fresh manifest/receipt. Prove exact output-only manifest derivation
and equivalent catalogue/native-cache semantics. Keep all 398 immediate-parent
PASS obligations, the cumulative 563-path ledger and every original failure.
This successor was discussed but no sealed executed family exists yet.

## Additional prepared work

Pending focused finalizers for rows18/19/20 are saved, unrun and guarded with
future proof fields NULL. Their independent reviews are respectively
`194ddc98`, `7d176481`, `1e882107`; recover exact paths under R. Rows16/17
pending finalizers were already reviewed as `400a7b45`. Later materializers and
scope recipes remain pending as recorded in the checkpoint. Do not borrow
another source's tests or audit to finalize them.

## Resources and coordination

The last paired release retained 10,376 recorded births:
`R/modularisation-L4-controls655-and16-gates-release-ROOT-ACTUAL-20261022/`.
Its independently reviewed union/absence/release reconstruction is
`R/modularisation-after04-paired-resource-orchestration-independent-SAVED-REVIEW-20261022/review.json`,
SHA `d8a6a2a1b689eb2aab969cc51e967f3b5c4ade6e4795e5baf53df0bf1573516c`.
The next union must append the entire failed controls family, completed step16
family, all later metadata actors and builders, retaining all seven PID-only
obligations. Use the existing reviewed tools and fresh absence checks.

Metadata gap `04101de6` remains explicitly unresolved: a parse-failed metadata
interpreter's birth was never observed. Preserve its synchronous completion
evidence and independent gap review `4278d1f6` separately; never invent an
identity or claim complete metadata birth capture. Fixed169 classification
`b6caef48` found no additional unmatched identities. New metadata wrappers
record their full own identity BEFORE compiling new source in the same process.

At handoff inspection both heavy-slot lock directories were absent and no
sub-agents remained in this agent's registry. These are observations, not fresh
resource authorization or a host-idle claim. Reconcile saved evidence first.
Use four agents total if supported and at most two heavy slots. Tutorial timeout
stays 300 seconds; component/facade/test limits and strict coverage stay fixed.
Do not weaken assertions. Run gates, commit and push as separate actions.
Only publish after the existing acceptance requirements pass; watch CI then.
