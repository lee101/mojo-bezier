"""Measured Mojo kernels against upstream bezier; run only via pixi run bench."""
import os, platform, sys, time
import bezier
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import mojo_bezier as mb

def best(fn, repeats=5):
    fn(); result = float("inf")
    for _ in range(repeats):
        start = time.perf_counter(); fn(); result = min(result, time.perf_counter() - start)
    return result
def row(name, ours, theirs):
    a, b = best(ours), best(theirs); ratio = b / a
    print(f"| {name} | {a*1e3:.3f} ms | {b*1e3:.3f} ms | {ratio:.2f}x {'faster' if ratio > 1 else 'slower'} |")
def main():
    degree, count = 12, 20_000; rng = np.random.default_rng(0)
    nodes = np.asfortranarray(rng.normal(size=(2, degree + 1))); values = np.asfortranarray(rng.random(count))
    ours, theirs = mb.Curve.from_nodes(nodes), bezier.Curve.from_nodes(nodes)
    print(f"machine: {platform.processor() or platform.machine()}, {platform.platform()}")
    print("| kernel | mojo-bezier | upstream bezier | ratio |\n| --- | ---: | ---: | ---: |")
    row(f"evaluate_multi degree {degree}, {count:,} values", lambda: ours.evaluate_multi(values), lambda: theirs.evaluate_multi(values))
    row(f"subdivide degree {degree}", ours.subdivide, theirs.subdivide)
    a = np.asfortranarray([[0., .25, .75, 1.], [0., 1., -1., 0.]])
    b = np.asfortranarray([[0., 1.], [.2, .2]])
    mo_a, mo_b = mb.Curve.from_nodes(a), mb.Curve.from_nodes(b)
    py_a, py_b = bezier.Curve.from_nodes(a), bezier.Curve.from_nodes(b)
    row("planar cubic-line intersection", lambda: mo_a.intersect(mo_b), lambda: py_a.intersect(py_b))
if __name__ == "__main__": main()
