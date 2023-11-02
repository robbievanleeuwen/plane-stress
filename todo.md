# TODO

## Pre-processor

- Plot BCs
- Some kind of info if points/lines are not found for BCs?
- Meshing options
  - Fields
  - Add option to force nodes and lines into the mesh see
    https://deltares.github.io/pandamesh/api/gmsh.html
  - Get mesh statistics
- Testing overlapping facets, other testing? (see deltares pandamesh)
- Persistent load case
- Node(?) & line load normal to curves

## Analysis

- Improve solver method
- All round performance improvements
- Add validation tests
- Add sparse matrices, numba, pypardiso

## Post-processor

- Stress
  - Get stress at points/along line (plot this?)
  - Get max/min stress
  - Plot certain materials
  - Stress histogram??
- Animations
