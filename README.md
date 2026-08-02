# mojo-bezier

`mojo-bezier` is a standalone Mojo port of the compute-heavy core of the
[Python `bezier`](https://pypi.org/project/bezier/) package: Bezier evaluation,
subdivision, degree elevation, and planar curve intersection. Its Python API is
intentionally shaped like upstream's `Curve` API for this covered subset.

## Install

```bash
pixi install
pixi run build
pixi run test
```

```python
import numpy as np
from mojo_bezier import Curve

curve = Curve.from_nodes(np.asfortranarray([[0., .5, 1.], [0., 1., 0.]]))
left, right = curve.subdivide()
points = curve.evaluate_multi(np.linspace(0., 1., 100))
crossings = curve.intersect(Curve.from_nodes(np.asfortranarray([[0., 1.], [.4, .4]])))
```

## Coverage

Covered: `Curve`, `from_nodes`, `evaluate`, `evaluate_multi`, `subdivide`,
`specialize`, `elevate`, `reduce_`, `locate`, numerical `length`, planar
`intersect`, and planar `self_intersections`. Intersections use control-polygon
subdivision followed by Newton refinement. Not covered: `Triangle`,
implicitization, curve/triangle intersection, non-planar intersections, and
degenerate / coincident-overlap handling. Inputs must be finite, two-dimensional
`float64` NumPy arrays in Fortran order when `copy=False`; implicit dtype
conversion is deliberately rejected at the FFI boundary.

## How it works

The Python layer owns NumPy arrays in the upstream Fortran-contiguous
`(dimension, degree + 1)` layout and sends their addresses through ctypes to one
Mojo shared library. The library does not allocate: evaluation and subdivision
write caller-provided buffers. Python owns the intersection work buffer; Mojo
uses it as a control-polygon stack, performs each split, and refines the
resulting parameter candidates with Newton iteration. Fixed work buffers fail
explicitly rather than silently dropping candidates.

## Benchmarks

Measured with `pixi run bench` on the machine reported by that command:

<!-- BENCHMARKS -->

machine: x86_64, Linux-6.8.0-136-generic-x86_64-with-glibc2.39

| kernel | mojo-bezier | upstream bezier | ratio |
| --- | ---: | ---: | ---: |
| evaluate_multi degree 12, 20,000 values | 1.063 ms | 1.234 ms | 1.16x faster |
| subdivide degree 12 | 0.019 ms | 0.006 ms | 0.31x slower |
| planar cubic-line intersection | 0.077 ms | 0.027 ms | 0.35x slower |

`evaluate_multi` uses SIMD batches with scalar tails. Intersection calls Mojo
for control-polygon subdivision and Newton refinement, retaining NumPy-owned
buffers across the FFI boundary. No GPU path is included: these benchmarked kernels are either
small and branch-heavy or insufficiently arithmetic-intensive after host/device
transfer, so a GPU path would lose at practical sizes. These are measured
results, not projections.

<!-- /BENCHMARKS -->

MIT.
