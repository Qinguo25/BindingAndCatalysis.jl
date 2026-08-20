# BindingAndCatalysis.jl repository instructions

[`Architecture.md`](../Architecture.md) is the sole canonical description of
the package architecture that is currently implemented. Read it before making
architecture-level changes, and update it in the same commit as any change to
module ownership, public contracts, solver routing, caches, or source layout.
Do not duplicate the architecture in this file.

[`docs/fiber_chamber_design.md`](../docs/fiber_chamber_design.md) defines the
fiber/chamber geometric contract and the stage 2/3 roadmap. It is a design
document: a capability is planned, rather than implemented, unless
`Architecture.md` says otherwise.

For the current one-dimensional SIMO implementation:

- `condition_method=:pair_memo_dag` is the production default;
- `condition_method=:suffix_dag` is the comparison backend;
- maintained keywords include `recompute`, `reltol`, and `abstol`;
- removed main-branch keywords must fail with their guided migration errors.

Run the relevant focused tests and the complete test suite after architecture
or solver changes. Keep current behavior and future goals explicitly separated
in both code comments and documentation.
