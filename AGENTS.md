# xrdroot — instructions for coding agents

## Goal

xrdroot is a pure-Python, NumPy-backed replacement for CERN ROOT. It reads and
writes ROOT files, does what ROOT's analysis classes do (histograms, fits, RooFit,
RooStats, HistFactory, TMVA, RDataFrame, RNTuple, graphics), and runs ROOT's own
programs without ROOT:

- **PyROOT scripts** run against `xrdroot.pyroot`, a ROOT-compatible namespace
  (`TFile`, `TH1F`, `TCanvas`, `gROOT`, `gStyle`, `RooFit`, `RDataFrame` …).
- **C++ macros** (`.C`, `.cxx`) are translated to Python by `xrdroot.cint` and
  run against the same namespace (`xrdroot run FILE.C`, `.x`, `gROOT.ProcessLine`).

"Full Pythonic parity" means every tutorial in ROOT 6.40.04's `tutorials/`
directory either produces ROOT's output (stdout, written ROOT files, images) or
is refused with a clear, full-sentence `UnsupportedFeatureError` because it is
genuinely out of scope (GUI widgets, Qt, SYCL, ACLiC-only code). Real ROOT is a
**validation oracle only**: it must never become a runtime dependency.

Well-maintained data-science libraries may be used: iminuit (Minuit2), SciPy,
scikit-learn, XGBoost, PyTorch, pyhf, matplotlib, plotly, pyvista. NumPy is the
only hard dependency. Anything else is an optional extra in `pyproject.toml`,
imported lazily, and refused with the `pip install` that fixes it when missing
(see `src/xrdroot/library.py` for the pattern).

## Status (2026-10-22; integration validation in progress)

Latest modularisation checkpoint: local branch
`refactor/modularisation-commits-20261021` contains six gated implementation
commits, ending at `4fe45f16` (persistence providers). Its clean committed tree
matches source `29f56c7a` across 1,484 files; record `aff94b1d` verifies source
bytes, kinds, modes, symlink text and Git blobs. The preceding commits remain
`720d178d`, `c0b9d09a`, `83dbc8f3`, `8fd86c90` and `14ecd34a`.
Internal refactoring keeps one project and adds `xrdroot.cppstd`; separate
Python distributions remain deferred. Development-only Python 3.10 `tomli`
is approved. Main and the committed tutorial baseline remain unchanged.

Fifteen ordered prefixes now pass their own six gates. Step 14's independent
audit `7f8a7a1b` accepts 33,876 checks: both modes record 9,668 passed, one
skip and 46 expected failures, with strict 100% line and branch coverage over
92,442 statements and 22,596 branches. Step 15 now passes its own six gates,
accepted by independent audit `55b92801` across 30,258 checks. Both modes
record 9,673 passed, one skip and 46 expected failures, with strict 100%
coverage over 92,536 statements and 22,596 branches. Source is unchanged,
owned processes are drained and slot 1 is released. The math prefix's first
four sequential comparisons have finished at
`runs/modularisation-04-L4-affected659-quiet4-FIRST-20261021`. Controller
68477 records all seven stages exit 0, unchanged source, drained processes and
both heavy slots released at 23:18:49 UTC. Release `d31bb21e` retained 10,263
births after a fresh 10,261-birth absence check, all seven PID-only obligations,
native/build and both-slot checks. Read-only readiness reported CPU speed limit
100%; settings are unchanged.
The first math recipe metadata phase stopped at a mismatched blocking-stage
name before producing recipes; its failure is preserved. A finite source
correction is being independently reviewed. No tutorial or manifest authority
was exercised by that failed preparation.

The persistence group's complete 593-path capture records 398 PASS, 56 DIFF,
59 FAIL, 74 UNSUPPORTED and six ORACLE-FAIL. All 266 immediate-parent PASSes
survive; thirteen older PASS obligations remain lost. Controls audit `450f5665`,
joined builder `2b419be1` and independent reconstruction `6c4af537` preserve
all original failures. The actual L3 ledger extension `c5547b42`, independently
reviewed as `9a29ef3d` across 3,625 checks, retains all nine historical views
and the 563-path PASS union without adding paths. Original captures keep their
original immutable ledger bindings. No tutorial, ordered-prefix or global
acceptance is granted by the local implementation commit.

Math's proposed 659-path scope includes every persistence path and the original
405 math paths; source review `6a6159e2` passes. Final capture-family review
`2544676b` passes 261 checks, retaining actual parent `4fe45f16`, all 398
observed parent PASSes, thirteen older losses and the nine-view ledger. Its
catalogue/cache preflight passes in 58.241 seconds with receipt `d041841a`;
genuine-client provenance passes. Independent saved audit `3f172a86` now passes 17,296 checks and accepts
capture integrity only. All four comparisons are FAIL: Python/C++ df103 and
hist103 time out at 300.193, 300.096 and 300.153 seconds; ntpl004 exits 2
without timeout after 6.983 seconds. All four earlier PASS obligations remain
open, without substitution or waiver. The controls recipe is now bound to
the actual receipt and audit (`5d2e7855`); independent output review `82549ddc`
passes 78 checks. A fresh resource union retains 10,374 recorded births and all
seven PID-only obligations; verification and release retain 10,376 births.
The paired release started controls655 and step16 gates in disjoint slots.
Controls controller 76032 stopped before any stage when its native-reader
guard observed the peer gate job's simultaneous `lsof` process. The failed
launch is preserved with exit 1, unchanged source, drained processes and slot 0
released. Step16 controller 76088 continues in slot 1; a controls retry requires
fresh resources and an honest failed-launch audit. Quiet-audit inputs v3 and launcher `771cfd98`
produced that saved audit after coordinator source review `cd7f3aa6` and
explicit ROOT authorization; their completion guards passed. A separate parse-failed metadata interpreter
has no observed PID or birth. Gap packet `04101de6` preserves its original
source, raw error, empty stdout and synchronous exit 1; no identity is invented
or full-birth/absence claim made. Corrected preparers record identity before
compilation. The next resource union must retain that gap separately.
Independent gap review `4278d1f6` passes 36 failure-evidence checks while keeping
the identity unresolved and granting no resource eligibility. The pending
controls concretizer passes source review `c63b2d2d`; its actual bound source
passes review `2289a692`. Post-quiet paired-plan review `aeb31fdf` passes 303
checks; fixed169 classification `b6caef48` finds no additional unmatched actor
identities. Gap `04101de6` remains separate and unresolved; no complete metadata
birth capture is claimed. Actual runtime proofs are saved separately from the
immutable source plan.
Pending focused finalizers for steps16/17 pass review `400a7b45`; future own
gate, release, summary and coverage fields remain NULL. No proof transfers. Proposed successor scopes
05–07 select 688, 688 and 730 paths; independent scope review `7452b031` and
pending-family review `bedfc2f2` pass, with runtime authority still false.
Conservative 893-path stdlib planning and the no-additional-path decision for
rows 23–25 retain every parent PASS obligation and the final full 1,189-path
ratchet. No evidence transfers between sources.

The separate complete candidate `7c01ca5b` passes all six gates: both test
modes record 9,771 passed, one skip, 46 expected failures and strict 100%
coverage across 92,632 statements and 22,598 branches. Independent audit
`69028760` accepts only that source. It does not replace any prefix's gates,
affected comparisons, final survey, wheel verification or tutorial ratchet.
Earlier details below are historical; use the latest durable coordinator
checkpoint for active jobs and acceptance state.

Shared-primitives candidate `6b5581d5` also passes all six gates, with 9,495
passed, one skip, 46 expected failures and strict 100% coverage in both modes
(audit `a849eee2`). Its quiet4 comparisons record two PASSes and two FAILs:
Python df103 times out at 300.043 seconds; ntpl004 receives invalid ROOT data.
All 293 controls now finished: 217 PASS, 19 DIFF, 23 FAIL, 32 UNSUPPORTED and
two ORACLE-FAIL. Audit `6b96f926` accepts 14,132 capture checks. Six controls
lose current-parent PASS: df014, df015, df029, df030 and both hist039
variants; their captures show failed data downloads with HTTP 401 responses.
Combined with quiet4, this is 219 PASS across 297 paths and eight strict
parent PASS losses. No scope or global acceptance is granted. The first
controls launcher and the split-preflight metadata failures remain preserved;
the corrected preflight passes audit `a6a4fbb0` across all 297 paths and 299
native captures.

Unchanged Python df103 rechecks time out at 300.049 and 300.039 seconds; the
accepted L9 parent also times out at 300.052 seconds. Their original PASS
captures remain intact. All diagnostic captures pass independent audits,
without a timeout waiver or performance claim. The bounded L1 profile retains
five checkpoints and a raw timeout at 300.085 seconds (audit `4aace1d9`);
network waits and overlapping compute costs establish no code cause. The
literal ntpl004 URL redirects to CERN authentication and HTML in the normal
client probe (audit `ad6b1fdb`). A separate native ROOT run of the unchanged
original macro exits 6 in 18.106 seconds: its original URL returns HTTP 401
and RDataFrame cannot open the file. Source and cleanup checks pass;
independent saved audit `9298f5d0` accepts 15,306 capture checks. No URL,
input, cache or comparison policy changes.

The first 25-prefix inventory batch completes 16 inventories, then exceeds the
unchanged 40-second checker bound at printf. Its independently audited failure
`3ebbcf2c` preserves the remaining eight unrun inventories and grants no
complete inventory acceptance. The fresh quiet SECOND batch finishes with all
77 stages passing, unchanged source, drained processes and both slots released
at 14:54:35 UTC. All 25 inventories are fresh; printf passes in 16.961
seconds. No first-batch results are pooled into it. Its released spec is
`4aa9faa8`, source-reviewed HELD `6b1555d0`, and release retains 6,223
completed process births. A pre-run read-only snapshot reports CPU speed limit
100%; previous snapshots at 53–55% establish no cause for earlier failures.
Power settings and limits are unchanged. Its first saved audit `a3e68e62`
stops after 14 checks because it tries to JSON-parse an empty retained raw
log. The failed audit and original source are preserved; byte-only validator
successor `6d1b4641` passes independent source review `5eb2856f` and actual
saved audit `aeabe3b7` with 123,872 checks. All 25 measured Counters add no
violations, including the first against L1 debt 58. Reviewed debt is 47, 41,
36, 32, 25, 12, 8, then seven through step 20, one at step 21 and zero from
session onward. Exact allowance approval `6284d71f` preserves the original
proposal. All 25 baselines are now materialized in separate worktrees; summary
`603327d7` records source-only success with no gate, runtime, ordered-prefix
or tutorial acceptance. The first two six-gate jobs have completed: 01-L7
controller 49620 in slot 0 and 02-L2 controller 49875 in slot 1. Their source
maps are `f226c8ef` (1,477 files, debt 47) and `0c684cec` (1,482 files, debt
41). Fresh resource verification passes for 6,348 retained completed births;
the paired release retains 6,350 and permits only these two jobs. Read-only
readiness at 15:19:29 UTC reports CPU speed limit 100%. Both runs retain
unchanged source, drained owned processes and released slots. Independent
saved audits `358496b2` and `c631d9c1` pass 25,781 and 25,808 checks. L7
records 9,495 passed in each mode with strict 100% coverage across 92,149
statements and 22,574 branches; L2 records 9,503 passed in each mode with
strict 100% across 92,180 statements and 22,578 branches. Each mode also
records one skip and 46 expected failures. Typing, Ruff, maintainability,
strict docs, compatibility and the unchanged 40-second component ratchet
pass. L7's affected captures and local implementation commit are now complete
as recorded below; L2 comparisons and its local commit remain pending. These
gates confer no tutorial or global-ratchet acceptance.
The post-gate resource union retains 6,597 completed births; fresh absence,
native/build boundaries and both free slots pass. The reviewed L7 inner
preflight manifest is concretely rebound (`fa5e39cc`) while its outer jobs
remain HELD. Idle-Mac readiness readings at 15:57, 16:01 and 16:03 UTC
report CPU speed limits of 62%, 34% and 48%; settings are unchanged and no
cause for historical failures is inferred. The user confirmed power and
ventilation were checked. A later read-only observation reports 51% with host
load around 15. After waiting, bounded L7 capture is proceeding under that
recorded load: controller 59909 at
`runs/modularisation-01-L7-affected321-quiet4-FIRST-20261021`, release
`9c28a236` with 6,609 retained births. Both heavy slots are reserved; controls
remain held. Any loaded timeout requires a quieter recheck before regression
attribution. No timing, tutorial, commit or global acceptance is claimed.
Its actual catalogue/cache preflight passes all 321 requested paths and the
323-path native dependency closure (`ce12ccba`). The completed four comparisons
are all FAIL: Python df103, C++ df103 and hist103 time out at 300.049,
300.060 and 300.073 seconds; ntpl004 exits 2 with a ROOT-format error in
5.670 seconds. These failures are not accepted regressions or waived PASS
obligations. Corrected saved auditor `854adbff` passes independent source
review `be49ef32`, including raw boundary proofs, and actual saved audit
`42d0582e` accepts 12,104 checks. Source is unchanged, owned processes are
drained and both slots are released. The audit accepts capture integrity only;
all four committed/accepted-L9 PASS losses and every scope/global hold remain.
The post-quiet resource union retains 6,697 births and passes fresh absence,
native/build, PID-only and slot checks. Paired release `cee4dd74`/`7c3c56f8`
retains 6,704 births. Read-only readiness at 16:38:44 UTC reports CPU speed
limit 100%. The persistence prefix's gates finish in slot 1, controller
65492 at `runs/modularisation-ordered-prefix-03-L3-all6-FIRST-20261021`.
Both modes record 9,536 passed, one skip and 46 expected failures, with strict
100% coverage across 92,240 statements and 22,596 branches. Independent audit
`cd33607e` accepts 26,189 checks and the source-specific six gates. The math
prefix then completes all ten stages with exit 0 at 17:26:57 UTC, controller
74670 at `runs/modularisation-ordered-prefix-04-L4-all6-FIRST-20261021`;
independent saved audit `3d01f570` accepts 30,855 checks and all six gates.
Both modes record 9,622 passed with strict 100% coverage across 92,255
statements and 22,596 branches. Source, owned-process drain and slot
release checks pass. Its guarded release `4d4e76e3` retains 6,842 completed
births and explicitly treats the live controls as a separate peer. The 317
controls then finish in slot 0, controller 65953 at
`runs/modularisation-01-L7-affected321-controls317-FIRST-20261021`, at
17:29:22 UTC: 217 PASS, 24 DIFF, 32 FAIL, 42 UNSUPPORTED and two
ORACLE-FAIL. Saved audit `dc69b047` accepts 17,412 capture checks and retains
nine control PASS losses. The first joined builder fails on missing policy
constants before producing a joined artifact (`1fe32d31`). Its narrow
successor preserves every function and assertion, passes static undefined-name
checks and independent source review `7835f472`, then completes saved joining
and independent reconstruction with 3,693 and 3,727 checks (`76c10db1`).
All 321 paths are captured: 217 PASS, 24 DIFF, 36 FAIL, 42 UNSUPPORTED and
two ORACLE-FAIL. All thirteen parent PASS losses remain open. A root-review
preparer's separate command-record schema error is also preserved; no science
was rerun for these metadata corrections. The attempted
next active-peer resource build stops before inventory because its peer has
completed; its failure is retained. The unchanged both-free successor retains
7,324 births and passes fresh absence, native/build, PID-only and slot checks.
Actual audits use each job's own source/process/slot completion. Gate recipes
for all remaining prefixes are prepared. The next pair completes:
05-L8 controller 81100 in slot 1 and 06-L5 controller 81199 in slot 0,
at their `runs/modularisation-ordered-prefix-*-all6-FIRST-20261021` paths.
Paired release `e08b995e`/`87ff21eb` retains 7,332 completed births. Independent
saved audits `3018ae98` and `47a05814` pass 26,880 and 26,934 checks. Each
test mode records 9,627 passed for L8 and 9,643 for L5, one skip and 46
expected failures, with strict 100% coverage across 92,301/92,343 statements
and 22,598 branches. All remaining gates and own lifecycle checks pass.
Affected tutorial acceptance remains pending. Later recipes remain held; extra zero-debt
checks occur only from the session prefix onward.
The independently reviewed copy-only decision (`fdfbe5e6`) and exact staged
index check pass. Local commit `8fd86c90` changes seven adapters plus the
component baseline, reducing measured debt from 58 to 47. It accepts only the
implementation tree; all thirteen tutorial obligations and the global hold
remain. Before L2 comparisons, an additional immediate-parent capture view is
being linked explicitly: all 217 L7 PASS paths are selected in L2 and already
protected by its older obligation union. No individual PASS is waived merely
because the whole parent scope remains unaccepted. Published branches and the
tutorial baseline are unchanged.
L2's explicit strict-parent sidecar and metadata auditor successor pass source
review `1428489c`; its preflight manifest, tutorial commands, selection and
comparison settings are unchanged. The fresh post-gate union retains 7,611
completed births and passes absence, native/build, PID-only and both-slot
checks. Read-only readiness reports CPU speed and scheduler limits of 100%.
Quiet release `f5d6d164` retains 7,615 births and launches controller 89393 at
`runs/modularisation-02-L2-affected356-quiet4-FIRST-20261021`, reserving both
heavy slots. Its actual catalogue/cache preflight passes for all 356 selected
paths and 358 native captures (`55c244c1`). All four comparisons finish FAIL:
Python/C++ df103 and hist103 time out at 300.032, 300.035 and 300.031
seconds; ntpl004 exits 2 with a format error after 4.314 seconds.
A separate post-df103 read-only snapshot still reports CPU speed and scheduler
limits of 100%, without a causal or host-idle claim. Concrete saved
audit inputs are sealed as `d0c2b8fc`; actual independent audit `778cbf26`
passes 14,596 capture checks with unchanged source, drained processes and
both slots released at 18:22:42 UTC. All four older PASS obligations remain.
The immediate L7 view also stays strict; those four paths were already FAIL
there, and its 217 PASS paths remain required in the controls. No
source-specific tutorial acceptance or global waiver is inferred. The prepared
07-L6 and 08-streams gate recipes stayed held throughout that quiet window.
The next pair began with L2 controls352 controller 94504 in slot 0 plus
07-L6 gates controller 94350 in slot 1. Their outputs are
`runs/modularisation-02-L2-affected356-controls352-FIRST-20261021` and
`runs/modularisation-ordered-prefix-07-L6-all6-FIRST-20261021`.
Controls binder `32f0d805`, reviewed as `9ef55562`, executes successfully;
bound HELD `4d1db194` preserves every failure. The fresh resource union
retains 7,694 births; absence, native/build, PID-only and slot checks pass.
Paired release `94ca4682`/`3601337e` retains 7,698 births. Read-only readiness
remains 100%. Actual audit inputs `02e1d91f`/`31c9a050` are bound and wait
for each job's own full capture/gates and lifecycle, with no peer-cleanup claim.
Both 07-L6 test modes now complete with 9,650 passed, one skip and 46 expected
failures, with strict 100% coverage across 92,355 statements and 22,598 branches.
Typing across 835 source files and strict documentation also pass. Independent
saved audit `bb5ae33e` accepts 27,303 checks for this source's six gates;
tutorial, commit and global acceptance remain false.
The streams/strings slot-1 plan passes source review `ed452d87`. After the
completed 07 capture, peer-aware resource build and fresh verification retain
7,825 completed births without removing the live L2 family from any prior list.
Release `fc611595` retains 7,830 births and starts streams/strings controller
1394 in slot 1 at `runs/modularisation-ordered-prefix-08-streams-strings-all6-FIRST-20261021`.
Both test modes complete with 9,655 passed, one skip, 46 expected failures and
strict 100% coverage across 92,369 statements and 22,598 branches. All six
gates pass; independent audit `92b9f7f9` accepts 32,013 checks for exact source
`acc5774a`, 1,510 files and debt seven. This accepts no tutorial scope.
The next resource union retains 7,975 completed births and passes fresh
peer-aware verification. Arithmetic release `3fc4e65c` retains 7,979 births;
controller 8498 runs in slot 1 at
`runs/modularisation-ordered-prefix-09-arith-cells-all6-FIRST-20261021`.
Its startup checks pass. L2 controls subsequently complete at 19:16:43 UTC,
with unchanged source, drained processes and slot 0 released. Independent
audit `a96a0a09` accepts 20,911 capture checks: 266 PASS, 23 DIFF, 23 FAIL,
38 UNSUPPORTED and two ORACLE-FAIL. All 217 immediate L7 parent PASSes survive.
Joining the original quiet4 gives 356 paths: 266 PASS, 23 DIFF, 27 FAIL,
38 UNSUPPORTED and two ORACLE-FAIL. Builder `60997898` and independent saved
review `0cf6a002` pass 4,112 and 4,146 checks. All thirteen older union-parent
PASS losses remain open. Copy-only decision `ece20c8c` passes independent
review `72795bd0`. Separate copy, staging and full index verification succeed;
the exact tree is committed locally as `14ecd34a`, with clean record `80ce91a3`.
All thirteen losses remain unresolved; no main publication is authorized.
Arithmetic completes at 19:28:10 UTC, with unchanged source, drained owned
processes and its slot released. Both modes pass 9,658 tests, one skip and
46 expected failures, with strict 100% coverage across 92,375 statements and
22,598 branches. Mypy checks all 840 modules. Independent audit `2debdeaf`
accepts 32,231 checks for exact source `6ec51477` (1,513 files, debt seven).
Nine ordered prefixes now pass their own six gates. Both preceding heavy jobs
are complete. Persistence source review `e0d110a2` passes 356 checks and binds
the actual L2 commit, all 266 immediate-parent PASSes and the cumulative 563
strict obligations. All 266 are selected; none adds a path outside that union.
Its effective jobs retain the original commands, settings and 300-second limit.
Fresh resource checks accept 8,522 completed births and both free slots.
Quiet release `a063177e` retains 8,540 births; controller 15099 reserved
both slots at `runs/modularisation-03-L3-affected593-quiet4-FIRST-20261021`.
Startup checks and actual catalogue/cache preflight pass for 593 selected
paths and 596 native captures (`71f8ab28`). All four original quiet captures
are FAIL: Python df103, C++ df103 and hist103 time out at 300.038, 300.042
and 300.041 seconds; ntpl004 exits 2 after 4.219 seconds on invalid ROOT data.
Independent audit `ed101084` passes 18,314 integrity checks and preserves all
four prior PASS obligations. Controls binding `a21c5b77` passes. Fresh checks
accept 8,619 known completed births; the paired release retains 8,621.
Step 10 then passes all six gates: both modes record 9,664 passed, one skip,
46 expected failures and strict 100% coverage across 92,405 statements and
22,598 branches. Mypy checks 844 files; audit `cf799368` accepts 28,391 checks.
Step 11 also passes all six gates: both modes record 9,664 passed, one skip,
46 expected failures and strict 100% coverage across 92,409 statements and
22,598 branches. Mypy checks 845 files; audit `099b208c` accepts 33,061 checks.
Eleven ordered prefixes now pass their own six gates. Controls589 remain active
in slot 0. Fresh peer checks retain 8,929 completed births and verify that exact
controls process separately; release retains 8,931 and starts step 12's gates
in slot 1 (`df008aab`). No peer absence or cleanup is claimed. CPU speed is 62%
and scheduler limit 100% at the latest fresh check; settings are unchanged.
Scope and global acceptance remain false.
Focused-basis source plans through step 25 pass independent review; future
execution references remain NULL. The stdlib materializer passes source review
`8cdc2641`, including guards for its two new directories, but its ordinary
tutorial execution remains held. A conservative 893-path native-backed
partition and separate 296 SKIPs pass source review `bc83034a`. All 48 sibling
directories are sealed after 359 additional files are hashed within existing
bounds; independent review `9fc5a23f` accepts that source evidence. Coordinator
selection `e308ade5` uses this conservative fallback for planning stdlib groups
08–22. Each still requires its own predecessor PASS evidence, extended ledger
and actual preflight proof. The 104-path discovery list is not approved scope.
The first finite recipe adapter stops on a metadata timeout-field assumption;
its failure is preserved. Corrected adapter source passes review `48212b21`;
ROOT derives fourteen held planning recipes, independently verified by
`b7cb5793` with 932 checks. No tutorial, oracle or gate runs during derivation.
All parent/capture/extended-ledger and runtime authority holds remain in place.
The final row-25 translator, full-harness and installed-wheel recipes pass
pending-source review `c1110cd8`. Their entry blocks prevent scientific dispatch
until separately reviewed successors bind the final state's own gates and all
required primary PASS evidence. No whole-candidate gate proof transfers to it.
At arithmetic release, readiness reports CPU speed 72% and scheduler limit 100%.
Power settings, limits and acceptance rules are unchanged. No peer cleanup,
host-idle or timing-improvement claim is made.
Finalized focused-basis records for 03 through 06
link their own completed full suites and coverage, preserving explicit limits
on per-node attribution. Their final reviews are `97cd6a8d` and `9a9a3f40`.
The finite 03 materializer `77bd93fb` passes source review `9f0b61ee` but
still needs the actual 593-path capture and review; the L2 commit now exists.
The finite 04 materializer also passes source review `8144a8ee`; its future
parent, capture and file-contract references remain pending.
The cumulative primary PASS ledger passes independent review `c3db5a30`:
563 strict paths retain committed, frozen-parent and earlier scope provenance,
including actual PASSes in unaccepted L1 and L7 captures. Later primary captures
still need adding. Finalized 08 focused evidence passes source review
`8c7bff50`, preserving all 34 cases and its attribution limits. Prepared finite
05 and 06 materializers pass source reviews `be3bf50c` and `9ec92c9f`; future
parent, capture and local-copy authorities remain held.
The finite 07 materializer passes source review `8f6a8518`; its future
authorities remain held. Finalized 09 focused metadata passes independent
review `0ebf62db`, preserving all fifteen cases and explicit attribution limits.
Later focused-basis drafts remain source-only until their own gates complete.
Use `followup_task` to restart a completed worker for a new task; `send_message`
only queues information and does not restart it.
The prepared final survey, wheel smoke and full-harness recipes are still
unrun. Before dispatch, bind them to the exact final committed source and its
own gates. The global ratchet must also retain every earlier captured PASS
obligation; the old 563-obligation proposal grants no waiver or automatic
acceptance of later parent captures.

The first saved joined297 builder fails on a metadata schema assumption
(`296e52d5`): the strict-gate audit records `passed`, while other audits
record `pass`. No joined artifact or downstream copy was produced. Narrow
corrected builder/reviewer `eba3d98a`/`16337079` pass independent source
review `dc0eb40a` and actual saved execution: 2,541 builder checks and 2,581
reviewer checks (`eb01f17c`). The eight losses and every scope/global hold
remain unchanged. The reviewed copy script `dc8a1960` then fails before any
copy because it also JSON-parses a raw test source. The failure remains
preserved with zero copied paths. Successor `b937d004` passes independent
review `0209de46` and the exact-source copy; local commit `83dbc8f3` then
preserves the six-gate tree and all eight comparison losses. The plan permits
a local implementation commit after its own gates and completed affected
comparisons; unresolved tutorial failures, definition-of-done item 7 and main
publication remain explicitly unaccepted. The third local implementation
commit is complete; the tutorial baseline and all published branches remain
unchanged.

A draft per-prefix saved-gate auditor `041ee177` was found to accept missing
child command proofs before any actual use. It and a failed source-only
fixture successor remain ineligible and preserved. Corrected source `27de2816`
passes 31 source-only negative/positive fixtures and passes independent source
review `76baf836`; its first two actual saved audits now pass as recorded
above, without inheriting historical acceptance. Every ordered prefix needs
its own six gates and affected
comparisons; whole-source gates do not transfer to those prefixes. The final
survey, full 1,189 tutorials, global ratchet and green CI remain required.
Original failures and all main-publication obligations persist.

The following paragraphs retain earlier source-specific history. Prefer the
latest checkpoint above and the authoritative durable handoff for current state.

The user has now authorized implementation of `MODULARISATION.md`: internal
refactoring in this single project, with `cppstd` as an additional component;
separate Python projects remain deferred. Refactoring uses accepted local
`8a008d88` in isolated `modularisation-*` worktrees. Its unchanged-source full
ordinary baseline finished at
`~/xrdroot-parity/runs/modularisation-8a008d88-ordinary-full-baseline-FIRST-20261021/`.
All three stages exit 0, the source is unchanged, and both heavy slots were
released. The record contains 558 PASS, 89 DIFF, 83 FAIL, 140 UNSUPPORTED,
296 SKIP and 23 ORACLE-FAIL. Its independent saved-data audit passes; five
committed PASS obligations remain lost: C++ df103 and hist103 timed out under
load, and Python rf102/rf404/rf406 retain their set-order DIFFs. No timeout
exception or global acceptance is inferred. The post-run resource-union builder
failed on a negative macOS host UID after the successful audit. A separate
metadata repair reused that audit, retained every UID and passed fresh absence
checks for 4,313 recorded births. A subsequent release sealed 4,315 births.
The original failure is preserved.
Source candidates now cover the facade, shared
primitives, display/persistence providers, engine maths, statistical/TMVA
adapters, direct engine imports, geometry and the generic C++ implementation.
They remain uncommitted. A separate source-only boundary preview
combines these candidates to find integration overlaps; it is not a gated tree.
Review found and corrected moved Jagged/STL provenance guards. The STL
compatibility path shares the generic module's actual global dictionary so
legacy attribute and dictionary rebinding still reach the implementation.
That candidate still needs runtime, coverage and tutorial validation.
The latest completed combined preview is source `cf6240b5` (1,557 files).
Its bounded strict component diagnostic reports 856 modules, 9,777 import
references and zero undeclared violations; the one exact Minuit contour private
import remains visible. Ruff passes, and the AST metric report over all 170
changed Python paths finds no violations across 2,565 functions. Source review
corrected provider-forwarder rebinding holes, and helper extractions preserve
assertions, rule ordering and the policy fingerprint. Earlier checker, Ruff,
binding-review and complexity findings remain retained with their own source.
These are source diagnostics, not runtime or six-gate acceptance.
The initial component inventory on restored source `acce6466` has an independent
eligibility receipt: 819 modules, 9,323 imports, 70 violations plus the visible
Minuit import. That unreviewed inventory is preserved unchanged. A tool edit
during review was caught, preserved separately and restored before the final
source check; its interrupted review is retained. The reviewed 70-debt initial
component baseline is materialized only in a separate candidate `f59f4ef0`;
the complete source's zero-debt baseline is materialized only in exploratory
candidate `51ec28fa`. Both source-only default ratchet checks pass. Two first
runtime gate attempts have now stopped with honest failures. The facade's cold
bootstrap passes, but forcing its original export order exposes the existing
writer/efficiency/hist/function circular import. The complete exploratory tree
passes strict/default component checks, Ruff and maintainability, then fails
mypy on 16 compatibility-shim re-export errors. No later stages ran in either
attempt. Both preserve unchanged source, drained owned processes and released
slots. Separate source successors are correcting these findings; no six-gate
or modularisation tutorial acceptance is claimed. Raw failures are durable in
`runs/modularisation-L9-facade-all6-FIRST-20261021` and
`runs/modularisation-whole-exploratory-all6-FIRST-20261021`.
The native baseline archive is now sealed at
`oracles/modularisation-parent-baseline-native-archive-FIRST-20261021`: 893 oracle
records and their outputs are independent copies with verified hashes; the
canonical cache is unchanged. A newer cleanup parser failure is retained; its
narrow adapter successor passes fresh absence checks for 4,359 recorded births.
The second release binds 4,361 births. The corrected facade source `7c6fca40`
passes the fresh compatibility probe and all six gates. Normal and forced OTHER
each record 9,351 passed, one skip, 46 expected failures and strict 100% coverage
across 92,133 statements and 22,574 branches; mypy checks all 819 source files.
Its independent saved-data audit passes 21,193 checks and proves all six gates.
Quiet4 records four PASSes, independently accepted by 8,064 capture checks.
Controls75 records 71 PASS and four pre-existing DIFFs; its independent audit
passes 8,984 checks with no current-parent PASS lost. Coordinator acceptance
`b59c3659` accepts only this 79-path scope and its gated source. The first
modularisation group is now committed locally as `720d178d` on
`refactor/modularisation-commits-20261021`; its clean committed tree matches
source `7c6fca40` exactly. Main and published branches are unchanged.

The complete exploratory source `34135bb7` passes strict/default component
checks, Ruff, maintainability and mypy across 856 modules. Its normal test run
records 9,734 passed, six failed, one skip and 46 expected failures, retaining
strict 100% coverage across 92,632 statements and 22,598 branches. OTHER and
docs did not run. Independent audit `a13a5c1f` accepts the honest failure
capture and cleanup; it does not accept the gates. The six failures concern
three new test expectations/setups: a more specific component rejection,
Python 3.14's builtin membership diagnostic, and a fixture that disables the
graphics draw hook. Three reviewed test-only corrections preserve or strengthen
all assertions, with every production/policy/baseline byte unchanged.

Corrected source `5ff9b15d` (1,557 files) passes focused corrections and all six
gates at `runs/modularisation-whole-testfix-focused-all6-FOURTH-20261021`.
Both test modes record 9,740 passed, one skip and 46 expected failures, with
strict 100% coverage across 92,632 statements and 22,598 branches. Independent
audit `512d0216` accepts 12,055 checks for this exact source. Its actual public
CLI passes all fifteen fresh-process component imports, but the cppstd test
scope rejects pytest's injected `_pytest.assertion.rewrite` import from a fixture.
The honest failure is retained at
`runs/modularisation-whole5ff9-public-component-CLI-SECOND-20261021`.
Both jobs are complete and their slots released. A narrow checker correction
is sealed as P0 `2ab71bc3` and whole `c71d5180`, changing only the isolation
tool and its regression tests. Independent source review `b837f2d5` passes;
all original assertions, production bytes, policy and baselines are preserved.
The exception permits only pytest's exact rewrite helper from genuine test
files, after explicit blocked-import checks. Whole `c71d5180` passes Ruff,
maintainability, all 134 focused checker tests and all fifteen public component
imports. Its actual cppstd scope now runs: 60 pass and one fails because its
singleton NumPy-array expectation differs between NumPy 1 and 2. Original
scalar-first production behavior is unchanged; a test-only correction is being
reviewed. Independent audit `c6fc353f` accepts 12,779 capture checks, retaining
the failure and unrun parity-after stage; it does not accept the whole source.
P0 `2ab71bc3` passes all six gates alone with both slots, controller 92541.
Both modes record 9,485 passed, one skip, 46 expected failures and strict 100%
coverage across 92,133 statements and 22,574 branches. The component ratchet
passes in 23.289 seconds within the same 40-second bound. Independent audit
`590b4709` accepts this exact source; its 819 production modules are byte-identical
to accepted L9, so this tool/test group adds no production-affected tutorials.
The old unrun audit's historical 9,323-import expectation was corrected in a
separate source: L9's four added imports make the authentic count 9,327.
After fresh 5,098-birth cleanup, P0 is committed locally as `c0b9d09a`, parent
`720d178d`; its clean committed tree matches all 1,474 gated files exactly.
Neither main nor any published branch changed.

Complete successor `ee093e1b` (1,557 files) combines the reviewed isolation tool,
the NumPy singleton-coercion test correction and approved conditional dev-only
tomli declaration. Independent source review `5a5e865b` accepts its exact bytes.
Its focused/public CLI run exits 0: 134 checker tests, all fifteen component
imports, all 61 isolated cppstd tests and both client provenance stages pass.
Source is unchanged and owned processes are drained. Independent saved audit
`8eeedaba` accepts 12,826 checks for these scoped proofs. Full six gates on this
successor stop honestly at maintainability in slot 1, controller 98069:
the new singleton-array assertion helper has CCN 11 against the unchanged limit
of 10. Earlier strict/default component checks and Ruff pass; later gates did
not run. Source is unchanged, owned processes are drained and slot 1 released.
Independent audit `0075257a` accepts 11,906 checks for the failure capture only.
The assertion-preserving helper split is now sealed as whole `7c01ca5b` (1,557
files), with independent source reviews `25390ffe` and `ec29249d`. Its expanded
three-file Ruff/maintainability checks, 134 checker tests, fifteen public
component imports and 61 isolated cppstd tests all pass; source is unchanged
and owned processes are drained. The saved audit and full six gates on this
successor remain pending. No gate acceptance transfers from `5ff9b15d`.
L1 successor `6b5581d5` passes all six gates: both modes record 9,495 passed,
one skip, 46 expected failures and strict 100% across 92,146 statements and
22,574 branches. Independent audit `a849eee2` accepts its exact source and
the genuine 821-module, 9,335-import, 58-debt inventory. Its four slow tutorial
comparisons now run sequentially with both heavy slots, controller 4498, at
`runs/modularisation-L1-toolfix-affected297-quiet4-FIRST-20261021`.
The fresh release binds 5,326 recorded births. The 293 controls and L1 commit
remain pending; the exact 297/299 native-cache preflight runs first. All 25
ordered source successors are sealed (`aa706da0`); five reviewed component
baselines are retained and the later twenty remain absent pending actual
inventories and independent review. Main, the committed tutorial baseline and
published branches remain unchanged.
Earlier corrected P0
source `b4dbfa92` stops during the component contract check at
`runs/modularisation-P0-cyclefix-ratchet-all6-SECOND-20261021`
(controller 80275). Its unchanged 40-second component bound expires under
concurrent load; no component result or later gate is captured. The source is
unchanged, owned processes are drained and slot 0 is released. Independent audit
`ec118064` accepts 11,572 timeout-capture checks only. The old unchanged retry
remains unrun; the active successor above includes the checker correction and
keeps the same 40-second bound with both slots reserved.
Fresh post-controls/THIRD cleanup verifies 4,742
recorded births absent; the release binds 4,744. Source-only propagation actor
79628 has finished its 25-prefix source-only propagation and remains tracked
for the next union. The new development-dependency preparation actor 82326
is separately tracked; no absence is claimed for ongoing source preparation.

The selected source sequence now covers all component lanes, cppstd helpers,
session adapters, pure tests, CI and docs. Source `2be18451` matches every byte
of complete `34135bb7` except its deliberately absent component baseline.
Separate successors propagate the three reviewed test corrections in order.
Seven earlier reviewed component inventories retain debt counts
70 → 58 → 47 → 41 → 36 → 32 → 25; later prefix inventories and gates remain
pending. The user has approved `tomli` as a development dependency only on
Python 3.10. CI/docs successors `6788532b`/`f9792b2f` add only that conditional dev
declaration; independent source review `3e122623` passes. Shared environments
are unchanged. The L1 affected scope is prepared and unrun: 297 tutorials and
299 genuine native dependency captures, without veto overrides. No complete modularisation
or global tutorial acceptance is claimed. The unvalidated reader candidate
remains separate. Current authority is in the latest durable checkpoint.

Local `main` is `e692236d8ff54916649cdcb613d36908bd30ba82`: accepted
Dawson compatibility (`486d2f89`), fitter-square (`30be2233`) and the xrdclient
`80677788` pin. Published `main` remains
`fa34ea9d4871b44cbbede35bf4f726b95c3c52ec`; its recorded
[CI](https://github.com/rob-c/xrdroot/actions/runs/39158140844) passed.
The approved integration branch `coordinator/tutorial-parity-r1-m2-20261020`
is at `e692236d`; all seven recorded
[CI jobs](https://github.com/rob-c/xrdroot/actions/runs/39409768861) passed.
The preceding `30be2233` CI failure, dependency versions and repair evidence
are retained. **Do not publish main until the global no-PASS-lost ratchet passes.**
Four separately gated fix groups are now committed locally on
`coordinator/tutorial-parity-speed-20261021`, ending at `8a008d88`:
`9c40e20f` (RDF worker caps), `737df4ad` (vector reads and row boxing),
`65952c59` (vector copy/assignment) and `8a008d88` (narrow return-copy elision).
Each complete committed tree matches its own accepted six-gate source prefix.
The branch is clean; neither main nor a published branch was updated.

Three full-harness records cover 1,189 tutorials against ROOT 6.40.04:

| Record | PASS | DIFF | FAIL | UNSUPPORTED | SKIP | HELPER | ORACLE-FAIL |
|--------|------|------|------|-------------|------|--------|-------------|
| Committed baseline | 558 | 87 | 69 | 134 | 326 | 0 | 15 |
| Recovery inventory, unaccepted | 572 | 91 | 116 | 235 | 42 | 0 | 133 |
| Modularisation parent, audited capture | 558 | 89 | 83 | 140 | 296 | 0 | 23 |

`tools/tutorials/baseline.json` is unchanged. The recovery inventory is
`~/xrdroot-parity/runs/full-inventory-recovery-prepared-20261018T220952Z/report/results.json`.
Its veto/oracle configuration differs; its counts are not accepted progress.
Targeted results do not replace either full record.

The inventory lost 12 committed PASSes. Four unchanged-main rechecks, two rf606
checks and vectorizedFit restored seven. Current073 now has separately accepted
latest scoped PASS evidence for df103 Python and ntpl004 C++; the full global
ratchet is still pending. The rf102/rf404/rf406 Python obligations remain open.
The original October 15
captures for the three set-order cases were lost in the OS crash; the user
confirmed no archive exists and retained the no-PASS-lost rule. Synthetic
set-order checks pass 3,012 source/lifecycle and 80,432 data checks for two
observed orders; this does not accept the three ordinary tutorials or change
input/comparison policy.

Current speed integration:

- The frozen current source is map
  `07316a850b055cc470886ed481cdeca936c7cd36d208f36f0c25fb5c13d89883`
  (1,457 files), in
  `/Users/rcurrie/src/xrdroot-wt/integration-CD-whole-vector-copyassignment-covfix-format-narrow-nrvo-after-pin806-20261020`.
  It combines C's vector-row work, D's thread caps, native-proven numeric and
  nested vector value/assignment corrections, and narrow named-local vector
  return elision. This exact source now forms the four local commits above;
  the frozen validation worktree remains preserved. Other prototypes stay separate.
- All six gates pass on this exact source. Normal and forced OTHER each record
  9,278 passed, one skip, 46 expected failures, and strict 100% coverage across
  92,130 statements and 22,566 branches. Ruff, maintainability, mypy across
  819 files and strict docs pass. Independent audit A3 (`b397a84e`) accepts
  2,821 checks, 35 client origins and 3,052 drained process births. The original
  audit's missing historical archive paths were repaired as metadata only;
  the failed audit is retained and science was not rerun for that repair.
- Focused validation passes 620 tests with strict helper coverage. The fixed
  720-sample NumPy 1/2 benchmark is independently accepted; all eligible row
  medians improve, while every fallback regression remains retained. These
  component timings establish no tutorial speed claim. Native9 NRVO and
  native28 assignment evidence are accepted for their recorded scopes.
  Linux327 is accepted on the preceding formatted source `80afc944`. The first
  current073 Linux405 protocol launcher stopped before any driver or tests:
  a duplicated check still expected 1,455 source files rather than the sealed
  1,457. The failed attempt is retained. Its fresh 17-check cleanup audit passes;
  the saved resource union now retains 3,234 births. A replacement that copied
  the client symlink as a directory was caught before execution and held.
  The symlink-preserving successor passes coordinator and independent source
  review. Its Linux unit and real observer protocols now pass, accepted by a
  fresh 69-check audit with 3,273 drained host births and remote cleanup verified.
  All 405 Linux numeric cases, including the 78 new NRVO cases, now pass.
  The independent 136-check audit accepts all 1,215 setup/call/teardown reports,
  verifies 3,307 drained host births, remote cleanup and slot 1 released. This
  supplements the current six gates; it does not accept the tutorial ratchet.
- The current ordinary 62-tutorial scope runs four quiet comparisons first,
  then 58 controls. The four finished sequentially with both heavy slots at
  `~/xrdroot-parity/runs/CD-current073-ordinary62-quiet4-FIRST-20261021`.
  Python df103, C++ df103 and hist103 are PASS in 270.305, 291.799 and
  264.984 seconds. Ntpl004 is FAIL: its HTTPS GET timed out after 142.404
  seconds, with raw exit 2 and `timed_out=false`. The independent 471-check
  audit accepts the honest capture and 3,121 drained births; whole-scope
  acceptance remains false and the ntpl004 PASS obligation is unchanged.
  All 58 controls pass at `runs/CD-current073-ordinary62-controls58-FIRST-20261021`;
  their independent audit accepts 705 checks and 3,199 drained births. The
  first joined62 view records 61 PASS, one FAIL and all 60 controls passing.
  Its independent saved-data review passes 389 checks. Ntpl004 is the sole
  selected baseline PASS loss, so scoped and global acceptance remain false.
  No baseline update or main publication is authorized by these results.
  One later isolated unchanged-source ntpl004 comparison passes in 58.275
  seconds. Its independent 426-check audit verifies the comparison, cache,
  source, 3,332 drained births and both slots released. This is retained as a
  separate diagnostic; the original first62 FAIL and its audit flags are unchanged.
  Coordinator decision `f8a649e6`, supported by independent saved review
  `d8af6224`, accepts the latest 62-path scope: three original quiet PASSes,
  58 control PASSes and the isolated ntpl004 PASS, at the exact same source,
  cache and comparison settings. This changes no original report, baseline,
  timeout exception or global-ratchet status, and applies only to current073.
  Previous separate diagnostics retain Python df103's 300.042-second timeout
  and the two C++ completions at 259.918/257.446 seconds.
- The bounded current073 Python df103 profile retains all five decoded
  checkpoints and a raw timeout at 300.046 seconds. Its 24-check lifecycle audit
  accepts diagnostic evidence and 3,087 drained births, with both slots released.
  The post-profile resource union retained 3,107 births; the post-quiet4 union
  retained 3,136. The post-controls/join union retained 3,216 births; the
  post-failed-launch union retains 3,234 after fresh cleanup validation.
  The profile confirms the old
  reconstruction-to-return-copy edge is gone. Scalar indexing and live namespace
  lookup remain costs; overlapping profile totals are not speed estimates.
- A separate lexical template-cache candidate is map `2a2c2867` (1,458 files).
  Its first 609-test run failed strict STL coverage at 99.02%; that failure is
  retained. Adding three existing test files, without source or assertion
  changes, produces 635 passed tests and strict 100% coverage across 600
  statements and 116 branches. Its independent audit accepts 146 checks,
  six client origins and 3,074 drained births. The first source draft's subclass
  StopIteration defect was caught before execution and repaired in this second
  draft. Fixed320 component timing, Linux9 and full-six-gate SOURCE chains are
  prepared or under review; none has executed or conferred performance acceptance.
  Linux9 is held: its duplicate launcher count still expects 1,455 rather than
  1,458 files, despite the corrected primary source guard. No Linux9 tests ran.
- The unchanged translator survey finished on current073 with exit 1 in
  36.786 seconds: 897 of 910 macros translated (98.6%), ten were refused and
  three crashed with invalid augmented assignments through `TTreeReaderValue`.
  The 98% floor is met, but the no-crashes gate fails. Raw outputs are durable
  at `runs/CD-current073-cint-survey98-FIRST-20261021`. The independent 16-check
  audit accepts the failure capture, verifies source and 3,241 drained births,
  and confirms slot 0 released; translator-gate acceptance remains false.
  The reader needs writable current-entry semantics as well as valid emitted
  stores; compiling an assignment into a temporary scalar array would lose writes.
  No source fix, refusal substitution or translator acceptance is claimed.
  The post-survey resource union retains 3,245 births, including the saved-union
  builder. The post-protocol union retained 3,287 births before numeric dispatch;
  the post-numeric union retained 3,321. The post-ntpl004 union now retains
  3,348. The post-native22 union now retains 3,363, including each builder.
  No current073 parity job is active; current modularisation jobs are listed above.
- A native ROOT fixture for writable `TTreeReaderValue` semantics has 22 fixed
  observation rows covering aliases, entry changes, arrays and integer narrowing.
  The native run exits 0 in 8.419 seconds and emits all 22 rows. Its independent
  65-check audit accepts the observations, verifies 3,357 drained births and both
  slots released. Shared branch buffers, lazy loads, same-entry preservation and
  narrowing are measured; direct branch-address retargeting is outside this
  fixture's scope. A reader/translator fix is being built in a separate worktree;
  it has not run tests or cleared the three survey crashes.
- The first reader candidate is frozen on clean parent `8a008d88`, map
  `630a1da8` (1,459 files; 13 changed paths). Independent source review passes
  28 checks. Initial static checks retain three honest failures: Ruff found one
  import-order error, maintainability found four metric violations across two
  inference functions and one regression test, and mypy found two untyped
  returns in the new reader buffer (820 files checked). The source remained
  unchanged and slot 0 was released. A separate successor is correcting these
  findings with all assertions preserved; no reader tests, coverage, full gates
  or tutorial acceptance are claimed. Evidence is durable at
  `recovery/CD-TTreeReader-initial-static-ROOT-20261021/`.
- The corrected reader candidate `c750e379` passes independent source review.
  Its first focused run records 533 passed and two failed in 26.66 seconds,
  with 91.09% coverage across all 11 requested modules. Both failures concern
  generated-code expectations; the new mutation test's runtime value/storage
  assertions pass. The run reports unchanged source, drained owned processes
  and slot 1 released. Its independent 131-check audit accepts the failure
  capture and 3,370 drained births while keeping strict scope false. The saved
  resource union retains 3,378 births, including the union builder.
  The test-only successor `a401636d` corrects the demonstrably wrong expectations,
  adds a float-pointee arithmetic regression and exercises public pointer errors.
  Independent source review passes; all production bytes remain exact to c750.
  Its full-six-gate setup is prepared and unrun. Dispatch is held because source
  review found that a reader load can replace an original C++ boolean variable
  with a NumPy scalar and break `std::boolalpha` output. An isolated successor
  restores the existing scalar normalization and adds a stream-format regression.
  The final `d7ef7dcd` candidate is sealed on `8a008d88` with 34 reader cases;
  its test restores the shared stream flag. Source review and the full-gate
  setup are reviewed. The first full-gate launcher stopped before creating a
  controller or running tests: its release guard received a null protocol path.
  Scheduling passed; launcher exit 1 and the traceback are retained at
  `recovery/CD-TTreeReader-mutable-lvalues-scalar-hygiene-all6-execution-FIRST-20261021/`.
  The source is unchanged; protocol binding and cleanup review remain pending.
  No coverage limit is lowered, and no reader gate or tutorial is accepted.
- A separate R2 invalid-supercategory-label fix passes independent source review
  and remains untested. It restores ROOT's missing diagnostic for an invalid label;
  it does not claim to resolve the earlier rf404/rf406 count or set-order DIFFs.

Earlier accepted R1 validation passed six gates and its 66-tutorial scope,
with all 60 controls passing; four loaded timeouts later passed isolated checks.
Fitter validation passed six gates and all 61 selected tutorials. The pinned
client passed both full test modes with strict 100% coverage. Earlier C/D,
combined-source and failed prototype evidence remains durable and attributed
to its exact source. Original failures are never overwritten or pooled into
current acceptance. Existing rf513 SIMCLONE gaps remain explicit.

Other lanes remain held or queued:

- Instrumentation 1–3 is published in `c6e14e4`: baseline/ratchet, timings and
  API coverage. The named `TTree::Draw("y:x")` graph fix is on main (`30f78c3`,
  `a7fc33d`); hist045 and gr302 pass both full records.
- Headless/helper harness changes passed six gates but await integration.
  The approved MPI command correction passed six gates; its controls retain
  a Bayesian Python timeout that also recurs alone within 300 seconds.
- Workspace persistence has 15 accepted native cross-reader checks and a
  730-test focused repair (three expected failures) with strict coverage across
  26 modules. Renewed full gates/native checks were interrupted by bounded
  process-monitor reads; those runs are unaccepted and writer execution is held.
- The likelihood candidate passes 333 focused tests with strict helper coverage
  and matches 174 native FCNs/statistics and RNG state. Timings do not establish
  improvement; it is uncommitted. The Poisson candidate's focused/native and
  480-sample evidence is retained, including regressions; Linux202 validation
  remains incomplete and full Bayesian validation is held.
- The second oracle is not globally ready. Its failing-build budget retains
  `397.1729134920025` seconds; preserve the durable ledger and approved options.

Use `~/xrdroot-parity/RECOVERY-STATE.json`'s `latest_coordinator_checkpoint`
for the authoritative handoff; older top-level active fields may be stale.
`STATUS.md`, `PLAN.md`, `runs/`, `analyses/` and `recovery/` retain full source
seals, historical failures, reports, oracle captures and process evidence.
Previous detailed AGENTS status is retained in the checkpoint snapshots and
`recovery/AGENTS-status-consolidation-ROOT-20261021/AGENTS.before.md`.
Store important and semi-critical material there before use. `/tmp` is disposable.
The current request uses four agents total: coordinator plus three workers.
The two-heavy-slot budget and unchanged 300-second tutorial limit still apply.

The gate environment is `.venv` (Python 3.14.8, NumPy 2.5.3). The original
`~/xrdroot-parity/venv312` (Python 3.12.14, NumPy 1.26.4) remains preserved.
Tutorials use `~/xrdroot-parity/venv312-pin806` with the genuine pinned client;
NumPy 1 tests use its separate `venv312-pin806-tests` environment. Shared
installations are unchanged. The user confirmed the Mac awake with its lid
open after thermal sleep events. Power settings, limits and acceptance rules
remain unchanged; fresh resource checks precede every execution.

## Where things are

| Area | Engine | ROOT-compatible layer |
|---|---|---|
| File I/O, trees, streamers | `src/xrdroot/{file,tree,streamers,writer,wtree,wbranch,…}.py` | `pyroot/trees`, `pyroot/core` |
| C++ → Python translator | `src/xrdroot/cint/` (`runtime/` is what translated code calls) | — |
| Histograms, graphs, functions | `hist.py`, `graph*.py`, `function/`, `formula/` | `pyroot/core`, `pyroot/graphics` |
| Fitting | `fit/` | `pyroot/core` |
| Canvas renderer (ROOT's pixels) | `canvas/` (rasteriser, fonts, TLatex layout) | `pyroot/graphics` |
| Plotting with matplotlib/plotly/bokeh | `plot/` | the draw hook in `pyroot/core` |
| RooFit / RooStats / HistFactory | `roofit/`, `roostats/`, `histfactory/` | `pyroot/roofit`, `pyroot/roostats` |
| RDataFrame | `rdf/` | `pyroot/rdf` |
| RNTuple | `rntuple/` | `pyroot/ntuple` |
| TMVA | `tmva/` | `pyroot/tmva.py` |
| Spectra (TSpectrum) | `spectrum/` | `pyroot/spectra` |
| Geometry (TGeo) | `geom/` | `pyroot/geom` |
| ROOT 7 (RHist, RCanvas) | — | `pyroot/histv7`, `pyroot/rcanvas` |
| Linear algebra | — | `pyroot/linalg` |

New PyROOT modules must be added to `SUBMODULES` in
`src/xrdroot/pyroot/__init__.py`. Follow how existing areas split an engine
package (plain Python API) from its `pyroot/` wrapper (ROOT's names and behaviour).

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,zstd]"          # plus extras: fit, tmva, geom, plot, …
export PYTHONPATH=$PWD/src            # always, especially in a git worktree
```

The **ROOT 6.40.04 oracle** is installed with micromamba from conda-forge
(Homebrew's ROOT does not install: its netpbm source is unreachable):

```bash
micromamba create -p ~/.local/root-oracle -c conda-forge root=6.40.04 python=3.12
```

The **tutorials** are a sparse clone of ROOT at the matching tag:

```bash
git clone --filter=blob:none --sparse --depth 1 -b v6-40-04 \
    https://github.com/root-project/root.git rootsrc-6.40
git -C rootsrc-6.40 sparse-checkout set tutorials
```

ROOT's own outputs are cached in `~/.cache/xrdroot-tutorials/oracle`, so ROOT
only reruns when a tutorial changes (`--refresh-oracle` forces a rerun).

## The tutorial harness: how progress is measured

```bash
python -m tools.tutorials run --tutorials rootsrc-6.40/tutorials --jobs 4 \
    --oracle-prefix ~/.local/root-oracle --xrdroot-src $PWD/src \
    --python ~/xrdroot-parity/venv312/bin/python \
    --out ~/xrdroot-parity/runs/full-UNIQUE-RUN-ID [--only roofit/roostats]
```

It interprets ROOT's tutorial `CMakeLists.txt`: which tutorials run, in what
order, and with what dependencies. It then runs each tutorial under ROOT and
under xrdroot, and compares stdout, written ROOT files and images. Each result
is PASS, DIFF, FAIL, UNSUPPORTED, SKIP or ORACLE-FAIL, written to
`results.json`, `summary.md` and `report.html` in `--out`.

- The per-tutorial timeout is 300 s. **Never raise it.** Make the code faster.
- Run long jobs (the full suite, a full harness run) as background scripts
  that write to a durable log file under `~/xrdroot-parity/`, so an interrupted
  session doesn't lose them.
- On a heavily loaded machine, timeouts are inflated. Rerun the slow ones alone
  before treating one as a regression.

The translator has its own ratchet:

```bash
python tools/cint_survey.py rootsrc-6.40/tutorials --min-percent 98
```

## Gates: every commit must pass all of these (CI runs them)

```bash
pytest -q -p no:cacheprovider -n 4 --cov=xrdroot     # 100% line AND branch coverage
XRDROOT_TEST_OTHER_MACHINE=1 pytest -q -p no:cacheprovider -n 4 --cov=xrdroot
ruff check src tests tools
mypy                                                 # strict
python tools/maintainability.py check
mkdocs build --strict
```

CI runs Linux on Python 3.10–3.13 and macOS on 3.12. Never chain gates, commit
and push with `;`: run the gates, then commit, then push, as separate steps.
After pushing, watch CI (`gh run watch`) and fix it until it's green.

Hard rules:

- **Never raise a limit.** The maintainability limits are CCN ≤ 10,
  cognitive ≤ 15, NPath ≤ 200, Halstead volume ≤ 500, nesting ≤ 4. Don't lower
  coverage, raise the harness timeout, or add `# pragma: no cover` to dodge a
  branch. Split functions instead.
- **Never delete or loosen an assertion to make it pass.** Find out whether the
  code or the test is wrong.
- **Numbers that must match ROOT exactly:**
  - ROOT's reference outputs were printed on darwin x86_64. `tests/refmachine.py`
    defines `ROOTS_MACHINE`: assert bit-exact there.
  - Elsewhere, use a tolerance with a comment explaining the numerics:
    `FIT_REL`, `PROFILE_REL` (1e-9), `NLL_REL` (1e-12), `roots()`. Typical
    causes are glibc vs Apple libm rounding (`lgamma`, `erf`), FMA on arm64, and
    `sum()` before Python 3.12.
  - Route transcendental functions through `xrdroot.random.libm` wherever ROOT
    exactness is claimed. NumPy's ufuncs use AVX-512 code paths on some CI
    runners and round differently from the C library ROOT calls.
- Tests that need an optional package must skip cleanly when it is missing.
- Python 3.10 is the floor; `xrdclient`, a hard dependency, needs it.
- pytest-timeout uses `timeout_method = "thread"`. The signal method
  deadlocked coverage's tracer lock and hung CI; don't change it back.

## House style

- Docstrings are prose explaining *why*, not restating the signature.
  Constants carry `#:` comments.
- Refusals are full sentences that say what is unsupported and what to do
  instead: `raise UnsupportedFeatureError("…")`.
- Test names are sentences:
  `test_a_url_is_fetched_whole_before_it_is_read`.
- Match the surrounding code's naming, density and idiom. Keep commits small,
  with messages in the existing style (`feat:`, `fix:`, `perf:`, `tests:`,
  `docs:`). Document new user-facing classes in `docs/root.md`.

## What remains, in priority order

Before you start, rerun the full harness on `main` as a baseline. Group every
FAIL, UNSUPPORTED and DIFF by root cause (the missing name or method, or the
failing feature) and count how many tutorials each cause blocks. Work from the
top, one gated commit per fix group, and rerun the affected tutorials after each.
Known items:

1. **Speed (≈24–40 timeouts).**
   - Mostly RooStats toy studies: FeldmanCousins, HypoTestInverter,
     StandardHypoTestDemo, One/TwoSidedFrequentistUpperLimitWithBands,
     rs101, rs701, StandardBayesianNumericalDemo.
   - Also: rf704, df012, df103, the hist103 NanoAOD analysis, ntpl016,
     the `mt_*`/`mp_*` ntuple fillers, fitFraction, minuit2FitBench.
   - Profile, then vectorise RooFit likelihood and toy generation. Keep ROOT's
     numerics: same integration rules, same random streams.
2. **`TTree::Draw("y:x")` must leave a `TGraph` named "Graph" in the pad**
   (hist045, gr302). Check the change doesn't alter images of tutorials that
   pass today.
3. **RooFit configuration and persistence.**
   - RooNumIntConfig / `defaultIntegratorConfig` (rf315, rf901, rf902).
   - `RooWorkspace::writeToFile` and ascii I/O (6 tutorials).
   - `RooLagrangianMorphFunc` (4).
   - RooSimWSTool SIMCLONE/EDIT (rf513), RooIntegralMorph (rf705), rf401/rf402.
   - The remaining DIFFs: rf211, rf316, rf608.py.
4. **RDataFrame.**
   - `Snapshot` of `vector<XYZTVector>` (df002).
   - `Book` (rf408).
   - String-valued `TTree::Draw` and RDF expressions (hist006, tree501).
   - df029, df030, df040.
5. **Maths.**
   - The remaining GenVector classes: rotations, Polar3DPoint, boosts
     (mathcoreGenVector).
   - GoFTest's Anderson–Darling test, matching ROOT (hist017, goftest).
   - TestBinomial; mathmoreIntegration and its multidim variant; pdf007.
   - `TF1Convolution`; `TMinuit` (the old interface, on iminuit).
6. **Missing classes**, each blocking a few tutorials:
   - `TFileMerger`, `TThreadExecutor`/`TThreadPoolTaskImp`, `TTask`.
   - `TMultiLayerPerceptron`, `THnSparseD`, `TSpider`, `fopen`.
   - `TAnnotation`, `gVirtualX` (batch no-ops), `ROOT.Numba`.
   - Classify each one: implement it, or refuse it in a full sentence.
7. **HistFactory and RooStats DIFFs.**
   - hf001's written file.
   - Zbi_Zgamma and HybridInstructional (factory `PROJ`).
   - Writing a `HypoTestInverterResult` to a file.
   - SPlot (rs301).
8. **Tree and ntuple output DIFFs.**
   - tree107, tree121, tree123, tree201.
   - ntpl007 and ntpl009 written files.
   - RNTuple fields of records, nested collections and low-precision floats.
   - `RNTupleImporter`.
9. **Graphics fidelity.** Most remaining image DIFFs are line widths, font
   metrics and text placement in `canvas/`. Fix them against ROOT's PNGs.
   Compare pixel by pixel where a test already does, as in
   `test_canvas_draw.py`.
10. **The last translator refusals.**
    - df018: a class template with a value parameter.
    - `math/r/Functor.C`: `ROOTR_EXPOSED_CLASS`.
    - view3ds.C: writing a character into a string in place.

## Parity instrumentation

Items 1–3 below are implemented in `c6e14e4`. Items 4–6 remain ongoing work;
existing cross-reader and platform checks cover only their recorded scopes.

1. **A committed tutorial baseline and ratchet.** Store a compact
   `tools/tutorials/baseline.json` (path → status). Add
   `python -m tools.tutorials ratchet RESULTS.json` that fails when a
   tutorial that passed is no longer PASS, unless the reason is a timeout
   under load. Optionally rerun those alone. Update the baseline in the same
   commit as the improvement.
2. **Per-tutorial timing.** Record wall time for both ROOT and xrdroot in
   `results.json`, and report the xrdroot/ROOT ratio and the slowest 30. Track
   the ratio over time so performance regressions show up before they become
   timeouts.
3. **API coverage against ROOT.** Generate the list of ROOT classes and
   public methods from the oracle install, either from its headers or from
   PyROOT's `dir()` on each class. Report what `xrdroot.pyroot` has, what it
   refuses and what is missing, by library (Core, Hist, Tree, RooFit, …). This
   is the parity measure for code beyond the tutorials.
4. **A file round-trip matrix.** For each class xrdroot writes, write a file
   and read it back with real ROOT, and with xrdroot from ROOT-written files,
   in a test that runs when the oracle is present and skips cleanly otherwise.
   Do this for every streamer layout: object branches, nested STL,
   HistFactory classes, RNTuple.
5. **Refusal inventory.** A script listing every `UnsupportedFeatureError`
   message in `src/` with the tutorials that hit it. This keeps the "out of
   scope" list explicit and reviewable.
6. **Cross-platform numerics checks.** Keep CI's macOS and Linux jobs. When
   adding an exact-value test, confirm it in Linux Docker (`python:3.x-slim`
   with CI's pinned versions) before pushing.

## Done means

- Every tutorial is PASS, or UNSUPPORTED with a full-sentence refusal of
  something genuinely out of scope (GUI, Qt, SYCL, ACLiC-only), or
  SKIP/ORACLE-FAIL for reasons outside xrdroot.
- The API-coverage report shows every non-GUI ROOT class used by the tutorials
  as implemented.
- All gates pass and CI is green on every platform.
