# TODO

## Pre-processor

- Plot BCs
- Meshing options
  - Get mesh statistics
- Testing overlapping facets, other testing? (see deltares pandamesh)
- Persistent load case
- Node(?) & line load normal to curves
- LineBC over multiple line tags?

## Analysis

- Improve solver method
- All round performance improvements
- Add benchmarks
- Add validation tests
- Add sparse matrices
  - What is the best way to modify values (BCs?)
  - Add force vector sparsity
  - Are list comprehensions the fastest?
- Add numba, pypardiso

## Post-processor

- Stress
  - Get stress at points/along line (plot this?)
  - Get max/min stress
  - Plot certain materials
  - Stress histogram??
- Animations
- Reactions
