# TODO

## Pre-processor

- Plot BCs
- Meshing options
  - Serendipity "Mesh.SecondOrderIncomplete"
  - Fields
  - Add option to force nodes and lines into the mesh see
    https://deltares.github.io/pandamesh/api/gmsh.html
- Testing overlapping facets, other testing? (see deltares pandamesh)
- Persistent load case
- Node(?) & line load normal to curves

## Analysis

- Add elements
  - Quad8
  - Quad9
- Improve solver method
- All round performance improvements
- Add validation tests
- Add sparse matrices, numba, pypardiso

## Post-processor

- Stress
  - Get stress at points
  - Get max/min stress
  - Principal stress & directions
  - Quiver plot
  - von Mises stress
  - Tresca stress
  - Plot certain materials
