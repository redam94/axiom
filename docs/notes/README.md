# docs/notes

Working notes: progress, changes, and decisions made while implementing the
plan in `docs/plan/`. One file per topic, numbered in the order opened.

- A **progress** note is a running log for a phase. It says what landed, what
  is left, and what was deliberately deferred.
- A **decision** note records a choice the plan left open, or a place where
  the implementation departs from the plan and why. `docs/plan/` is updated to
  match once the decision is stable; until then the note is authoritative.
- A **deviation** note (required by the porting protocol, CLAUDE.md §6) records
  a deliberate change to a number ported from `mmm-framework`, and is linked
  from the golden fixture's `superseded_by`.

| # | Note | Kind |
|---|---|---|
| 0001 | [Phase 1a — foundation progress](0001-phase-1a-progress.md) | progress |
| 0002 | [Decisions taken while building the foundation](0002-foundation-decisions.md) | decision |
| 0003 | [Where axiom deliberately differs from mmm-framework](0003-deviations-from-parent.md) | deviation |
| 0004 | [HYPER-3: a sequential-trial case study, and the module it exposed](0004-case-study-hypertension.md) | progress + decision |
| 0005 | [Basis response families: polynomial, spline, piecewise linear](0005-basis-response-families.md) | decision |
| 0006 | [Gaussian-process surfaces, and the recovery world that was never identified](0006-gaussian-process-surfaces.md) | decision + deviation |
| 0007 | [A surface never travels without its uncertainty, and the reports that carry it](0007-uncertainty-and-reports.md) | decision |
| 0008 | [PyMC as a third backend, and the PyTensor interpreter under it](0008-pymc-backend.md) | decision |
| 0009 | [TUTOR-60: one decision end to end, and the two io bugs it found](0009-case-study-tutoring.md) | progress + deviation |
| 0010 | [Systems that are not DAGs: simultaneity, time, and the compiler that removes both](0010-dynamic-systems.md) | decision |
| 0011 | [Identifiability of nonlinear parameters: which combination, not whether](0011-identifiability-of-combinations.md) | decision |
| 0012 | [Beyond DAGs: which alternative frameworks axiom implements, and which it does not](0012-beyond-dags.md) | decision |
| 0013 | [What the data can orient, and at what granularity](0013-discovery-and-granularity.md) | decision |
| 0014 | [Refuting the graph, resampling the search, and dropping sufficiency](0014-refutation-stability-and-latents.md) | decision |
| 0015 | [The report generator the charter refuses, as a package beside it](0015-report-narration-addon.md) | decision |
| 0016 | [Showing a result, and drawing the structure](0016-display-and-plots.md) | decision |
| 0017 | [A printed number stops where its uncertainty stops](0017-significant-digits.md) | decision |
| 0018 | [GEIGER-1911: designing an experiment, and what robustness costs](0018-case-study-rutherford.md) | progress |
| 0019 | [Writing an adapter: the seam, the five parts, and where things actually go](0019-writing-an-adapter.md) | decision + progress |
| 0020 | [The tutorial: one decision across every phase, ending in a document](0020-the-tutorial-notebook.md) | progress + decision |
| 0021 | [The HTML report is laid out as a paper, not as a page of defaults](0021-the-html-report-as-a-paper.md) | decision |
| 0022 | [A readout carries the run's own tables and charts, not a summary of them](0022-the-walkthrough-readouts.md) | decision + deviation |
| 0023 | [Each section is shown only the facts it may state](0023-what-each-section-may-say.md) | decision |
| 0024 | [A model reads like the mathematics, and builds the same tree](0024-expression-operators.md) | decision |
| 0025 | [A notebook is not a paler medium than a terminal](0025-notebooks-render-themselves.md) | decision + progress |
| 0026 | [A notebook that only demonstrates the API has not made the case for it](0026-a-notebook-has-to-argue-for-itself.md) | decision + progress |
| 0027 | [The invariants hold inside one analysis and stop at its edge](0027-scope-and-the-experiment-lifecycle.md) | decision + progress |
| 0028 | [Naming a bias is not correcting it](0028-the-estimate-a-stopped-study-may-report.md) | decision |
| 0029 | [Twelve studies from three clients are not twelve draws](0029-the-party-a-study-came-from.md) | decision |
| 0030 | [A well-diagnosed wrong number](0030-the-experiment-that-was-actually-run.md) | decision |
| 0031 | [Assigned is not received, and the readout has to say which](0031-assigned-is-not-received.md) | decision |
| 0032 | [A population you can size and cannot list](0032-a-population-you-can-size-and-cannot-list.md) | decision + deviation |
| 0033 | [Colliding is not automatically wrong, and the calendar knows which](0033-two-experiments-in-the-same-markets.md) | decision |
| 0034 | [A spending function cannot price a look nobody scheduled](0034-the-look-nobody-scheduled.md) | decision |
| 0035 | [Eight wrong go-decisions a quarter, all individually defensible](0035-eight-wrong-go-decisions-a-quarter.md) | decision |
| 0036 | [Silence is not agreement](0036-what-this-party-means-by-conversion.md) | decision |
| 0037 | [Reached is not observed, and a bound is not a downgraded point](0037-reached-is-not-observed.md) | decision |
