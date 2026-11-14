# Learn while implementing

**Git vs GitHub:** Git records source changes locally in commits. A branch permits isolated development. GitHub hosts the repository, pull-request review, and CI checks.

**Why CPU CI for GPU research?** Errors in configuration, split logic, labels and metrics can be detected without running a costly model. This prevents invalid experiments before training starts.

**Why not store model weights or raw datasets in Git?** Huge binary objects are difficult to review and can contain private or licensed data. Track recipes, source revisions, hashes and research outputs in appropriate controlled storage.

**Provenance manifest:** A JSON record of configuration, source commit, model ID/revision, dataset checksums and installed packages. A registered manifest does not establish that an experiment ran.

**Dataset leakage:** If the same network episode appears in both training and testing, the model may memorize circumstances rather than generalize. Every counterfactual rewrite of an episode must remain within its assigned split.

**Baseline and ablation:** A baseline is an established comparison method; an ablation removes one proposed contribution to determine if it independently helps.
