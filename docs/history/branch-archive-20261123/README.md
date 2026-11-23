# Branch archive, 2026-11-23

Every local branch that carried commits patch-distinct from `main` when the branches were
audited, as `git format-patch` output, one directory per branch. The patches are here so that
deleting a branch ref destroys nothing: `git am <dir>/*.patch` on a worktree off the commit the
set was taken from rebuilds it.

| Branch | Ahead | What it was |
|---|---|---|
| `r41-02` | 3 | feat: QuasiRandomSobol, QuasiRandomNiederreiter and RandomMT, by ROOT::Math's names |
| `r36-03` | 3 | docs: measure why the rmath one-liner is not a fix (r36-03) |
| `r37-01` | 2 | tests: a pointer range end through at(), and the algorithms over a container in process |
| `r36-02` | 2 | tests: the free Add and the in-place TMatrix members, against the numbers ROOT's Kalman benchmark prints |
| `r34-01` | 2 | tests: a clone's test writes its files under the tmp directory |
| `r28-01` | 2 | Revert "docs: a cloned TRef names the source file's ProcessID record" |
| `r27-05` | 2 | feat: a tree branch of whole TRef objects, as tref_test.C writes one |
| `r15-05` | 2 | fix: a tree read from a file is no object of memory to write back |
| `gup-01` | 2 | test: a session starts with no include path for the next test to inherit |
| `g11-alt-flat` | 2 | fix: Alt$ reads a two-dimensional branch flat and an indexed one only where it points |
| `wip/r68-tprofile2poly` | 1 | perf: a profile reads what it fills into once, and a poly finds its bin by box |
| `salv-a4fix` | 1 | salvage: TClass reflection dictionaries, minus the two hunks main already has |
| `remove-root-timing-comparisons-previous` | 1 | tools: no ROOT-to-xrdroot timing comparison, only xrdroot's own wall times |
| `ralph-round-202611230039-main` | 1 | feat: an axis' bin labels are written, in the layout ROOT writes them |
| `r9-05` | 1 | feat: a class the file records under another checksum is read as the program has it |
| `r8-09` | 1 | feat: ROOT.SetHeuristicMemoryPolicy is ROOT's deprecated switch, off by default and answering what it was |
| `r5-16` | 1 | docs: judging says what a WILLFAIL test now does |
| `r5-08` | 1 | feat: a ROOT collection read back is the namespace's collection, not the list it read as |
| `r41-03` | 1 | fix: a symmetric SMatrix's sum, difference and scale keep their cells |
| `r41-01` | 1 | feat: a member of a class that is one of ROOT's collections |
| `r4-05` | 1 | feat: a TRef starts with kIsOnHeap\|kNotDeleted and names its action in the bits |
| `r37-03` | 1 | feat: ROOT's own classes are the ones ROOT's headers declare |
| `r34-04` | 1 | feat: ROOT::Fit's Chi2FCN and ROOT::Math's GeneticMinimizer, the start genetic-GAMinTutorial and -testGAMinimizer fit from |
| `r30-04` | 1 | docs: the io/json TypeErrors are the translated char[N][M] constructor |
| `r30-01` | 1 | docs: what a TObject's fBits and fUniqueID are where ROOT prints them |
| `r26-04` | 1 | docs: a nested C++ type name reaches Python without its template arguments |
| `r22-02` | 1 | docs: the TArray arrays stream themselves, so a catalogue gives them no plan |
| `r19-04` | 1 | fix: a member of ROOT's read by the getter without a Get, as TObjArray::LowerBound |
| `r15-08` | 1 | perf: a record of numbers goes out in one struct and comes back into its members |
| `r15-07` | 1 | fix(rootsuite): leave a test's output log where its siblings read it |
| `r14-06` | 1 | fix: the build tree's <name>.log, so roottest's sibling log checks run |
| `r12-02` | 1 | feat: TGeoManager::Export writes a geometry back as the ROOT file it came from |
| `r11-15` | 1 | feat: one of ROOT's arrays of numbers is written as its count and values |
| `r11-05` | 1 | feat: a TArrayF or TArrayI member of a declared class is described as ROOT describes it |
| `archive/unjudged-salvage` | 1 | archive: the diffs rescued from abandoned worktrees, as patches |
