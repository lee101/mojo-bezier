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
| evaluate_multi degree 12, 20,000 values | 1.411 ms | 1.516 ms | 1.07x faster |
| subdivide degree 12 | 0.010 ms | 0.008 ms | 0.76x slower |
| planar cubic-line intersection | 0.028 ms | 0.034 ms | 1.22x faster |

`evaluate_multi` uses SIMD batches with scalar tails. Subdivision now uses an
in-place SIMD de Casteljau pass over caller-owned output storage, also with a
scalar tail. Intersection performs control-polygon subdivision, Newton
refinement, deduplication, and ordering in one Mojo call while reusing
thread-local NumPy-owned work buffers across the FFI boundary.

No parallel path is included: this pinned standard library does not expose
`parallelize`, and the benchmarked subdivision and intersection calls are too
small and dependent or branch-heavy to benefit from thread launch overhead. No
GPU path is included: the device-memory check passed with 12,320 MiB free, but
these kernels are either small and branch-heavy or below roughly two flops per
byte once control-node, value, result, and host/device traffic is counted. A GPU
path would therefore lose at the practical sizes covered here. These are
measured results, not projections.

<!-- /BENCHMARKS -->

MIT.
