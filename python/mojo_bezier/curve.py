"""Focused upstream-shaped ``bezier.Curve`` API backed by Mojo kernels."""
from __future__ import annotations
import enum
import threading
import numpy as np
from ._lib import addr, f64_fortran, lib


def _f64_values(value, *, name):
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64; implicit conversion is not supported.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite.")
    return np.ascontiguousarray(array).reshape(-1)

class IntersectionStrategy(enum.Enum):
    GEOMETRIC = 0
    ALGEBRAIC = 1

def _split(nodes, s):
    if s == 0.5:
        left, right = np.empty_like(nodes, order="F"), np.empty_like(nodes, order="F")
        lib().mbz_subdivide(addr(nodes), nodes.shape[0], nodes.shape[1] - 1, addr(left), addr(right))
        return left, right
    work = nodes.copy(order="F")
    left, right = np.empty_like(nodes, order="F"), np.empty_like(nodes, order="F")
    left[:, 0], right[:, -1] = work[:, 0], work[:, -1]
    for level in range(1, nodes.shape[1]):
        work[:, :-level] = (1.0 - s) * work[:, :-level] + s * work[:, 1:nodes.shape[1] - level + 1]
        left[:, level], right[:, -level - 1] = work[:, 0], work[:, nodes.shape[1] - level - 1]
    return left, right

def _evaluate(nodes, s):
    work = nodes.copy(order="F")
    for width in range(nodes.shape[1] - 1, 0, -1):
        work[:, :width] = (1.0 - s) * work[:, :width] + s * work[:, 1:width + 1]
    return work[:, 0]

def _derivative(nodes, s):
    degree = nodes.shape[1] - 1
    return np.zeros(nodes.shape[0]) if degree == 0 else _evaluate(degree * (nodes[:, 1:] - nodes[:, :-1]), s)

_intersection_storage = threading.local()

def _intersection_buffers(record, max_frames, max_pairs):
    required = max(max_frames * record + 8, max_pairs)
    work = getattr(_intersection_storage, "work", None)
    pairs = getattr(_intersection_storage, "pairs", None)
    if work is None or work.size < required:
        work = _intersection_storage.work = np.empty(required, dtype=np.float64)
    if pairs is None or pairs.shape[1] < max_pairs:
        pairs = _intersection_storage.pairs = np.empty((2, max_pairs), dtype=np.float64, order="F")
    return work, pairs

def _intersections(a, b):
    max_frames, max_pairs = 64, 16_384
    controls = 2 * (a.degree + b.degree + 2)
    record = controls + 5
    work, pairs = _intersection_buffers(record, max_frames, max_pairs)
    written = lib().mbz_intersect(
        addr(a.nodes), a.degree, addr(b.nodes), b.degree,
        addr(work), max_frames, addr(pairs), max_pairs,
    )
    if written < 0:
        reason = "candidate buffer" if written == -2 else "subdivision stack"
        raise RuntimeError(f"Intersection {reason} exhausted; the operation was not completed.")
    return pairs[:, :written].copy(order="F")

class Curve:
    @classmethod
    def _from_computed(cls, nodes, degree):
        curve = object.__new__(cls)
        curve._nodes, curve._degree = nodes, degree
        return curve

    def __init__(self, nodes, degree, *, copy=True, verify=True):
        array = f64_fortran(nodes, copy=copy, name="Nodes")
        if array.ndim != 2: raise ValueError("Nodes must be a 2D array.")
        if degree < 0 or array.shape[1] != degree + 1: raise ValueError("The number of nodes must equal degree + 1.")
        if array.shape[0] == 0: raise ValueError("Nodes must have at least one dimension.")
        if verify and not np.isfinite(array).all(): raise ValueError("Nodes must be finite.")
        self._nodes, self._degree = array, int(degree)
    @classmethod
    def from_nodes(cls, nodes, copy=True):
        array = np.asarray(nodes)
        if array.ndim != 2: raise ValueError("Nodes must be a 2D array.")
        return cls(array, array.shape[1] - 1, copy=copy)
    @property
    def nodes(self): return self._nodes
    @property
    def degree(self): return self._degree
    @property
    def dimension(self): return self._nodes.shape[0]
    def __repr__(self): return f"<Curve (degree={self.degree}, dimension={self.dimension})>"
    def evaluate(self, s): return self.evaluate_multi(np.asfortranarray([s]))
    def evaluate_multi(self, s_vals):
        values = _f64_values(s_vals, name="s_vals")
        result = np.empty((self.dimension, values.size), dtype=np.float64, order="F")
        lib().mbz_evaluate_multi(addr(self.nodes), self.dimension, self.degree, addr(values), values.size, addr(result))
        return result
    def subdivide(self):
        left, right = _split(self.nodes, 0.5)
        return self._from_computed(left, self.degree), self._from_computed(right, self.degree)
    def specialize(self, start, end):
        if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0: raise ValueError("Specialization parameters must be in [0, 1].")
        if start == end: return Curve(np.repeat(self.evaluate(start), self.degree + 1, axis=1), self.degree, copy=False)
        if start > end: return self.specialize(end, start).reverse()
        _, tail = _split(self.nodes, start)
        section, _ = _split(tail, (end - start) / (1.0 - start) if start != 1.0 else 0.0)
        return Curve(section, self.degree, copy=False)
    def reverse(self): return Curve(self.nodes[:, ::-1], self.degree)
    def elevate(self):
        result = np.empty((self.dimension, self.degree + 2), dtype=np.float64, order="F")
        lib().mbz_elevate(addr(self.nodes), self.dimension, self.degree, addr(result))
        return Curve(result, self.degree + 1, copy=False)
    def reduce_(self):
        if self.degree == 0: return None
        degree, elevated = self.degree - 1, self.nodes
        reduced = np.empty((self.dimension, degree + 1), dtype=np.float64, order="F")
        reduced[:, 0], reduced[:, -1] = elevated[:, 0], elevated[:, -1]
        for j in range(1, degree):
            alpha = j / (degree + 1)
            reduced[:, j] = (elevated[:, j] - alpha * reduced[:, j - 1]) / (1.0 - alpha)
        candidate = Curve(reduced, degree, copy=False)
        return candidate if np.allclose(candidate.elevate().nodes, elevated, rtol=0.0, atol=2e-12) else None
    @property
    def length(self):
        roots, weights = np.polynomial.legendre.leggauss(16)
        return float(0.5 * np.dot(weights, [np.linalg.norm(_derivative(self.nodes, 0.5 * (x + 1.0))) for x in roots]))
    def locate(self, point):
        point = _f64_values(point, name="point")
        if point.size != self.dimension: raise ValueError("Point dimension does not match curve dimension.")
        grid = np.linspace(0.0, 1.0, 257); values = self.evaluate_multi(grid)
        s = float(grid[np.argmin(np.sum((values - point[:, None]) ** 2, axis=0))])
        for _ in range(20):
            value, derivative = self.evaluate(s)[:, 0], _derivative(self.nodes, s)
            denom = float(derivative @ derivative)
            if denom == 0.0: break
            ns = min(1.0, max(0.0, s - float((value - point) @ derivative) / denom))
            if abs(ns - s) < 1e-14: s = ns; break
            s = ns
        if np.linalg.norm(self.evaluate(s)[:, 0] - point, ord=np.inf) > 1e-8: raise ValueError("Point is not on the curve.")
        return s
    def intersect(self, other, strategy=IntersectionStrategy.GEOMETRIC, verify=True):
        if not isinstance(other, Curve): raise TypeError("Can only intersect another Curve.")
        if self.dimension != 2 or other.dimension != 2: raise NotImplementedError("Intersection is currently implemented for planar curves.")
        if strategy not in (IntersectionStrategy.GEOMETRIC, IntersectionStrategy.ALGEBRAIC): raise ValueError("Unknown intersection strategy.")
        return _intersections(self, other)
    def self_intersections(self, strategy=IntersectionStrategy.GEOMETRIC, verify=True):
        if self.dimension != 2: raise NotImplementedError("Self intersection is currently implemented for planar curves.")
        left, right = self.subdivide(); pairs = left.intersect(right, strategy=strategy, verify=verify)
        if pairs.size == 0: return pairs
        pairs[0] *= 0.5; pairs[1] = 0.5 + 0.5 * pairs[1]
        return pairs[:, np.abs(pairs[0] - pairs[1]) > 1e-7]
