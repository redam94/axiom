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
