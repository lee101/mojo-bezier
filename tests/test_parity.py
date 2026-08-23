import numpy as np
import pytest
import bezier
import mojo_bezier as mb

CURVES = [
    np.asfortranarray([[0., .25, .8, 1.], [0., 1., -.2, .5]]),
    np.asfortranarray([[-2., 0., 1.], [1., -1., 2.]]),
    np.asfortranarray([[0., 1.], [2., -1.], [1., 3.]]),
]

@pytest.mark.parametrize("nodes", CURVES)
def test_constructor_properties_and_evaluation_match_upstream(nodes):
    ours, theirs = mb.Curve.from_nodes(nodes), bezier.Curve.from_nodes(nodes)
    values = np.asfortranarray(np.linspace(0., 1., 51))
    assert repr(ours) == repr(theirs)
    assert ours.degree == theirs.degree and ours.dimension == theirs.dimension
    assert np.allclose(ours.evaluate_multi(values), theirs.evaluate_multi(values), atol=2e-14)
    assert np.allclose(ours.evaluate(.371), theirs.evaluate(.371), atol=2e-14)

def test_evaluate_multi_simd_tail_and_large_batch_match_upstream():
    nodes = CURVES[0]
    ours, theirs = mb.Curve.from_nodes(nodes), bezier.Curve.from_nodes(nodes)
    values = np.asfortranarray(np.linspace(0., 1., 262_147))
    assert np.allclose(ours.evaluate_multi(values), theirs.evaluate_multi(values), atol=2e-14)

@pytest.mark.parametrize("nodes", CURVES)
def test_subdivide_specialize_and_elevate_match_upstream(nodes):
    ours, theirs = mb.Curve.from_nodes(nodes), bezier.Curve.from_nodes(nodes)
    ol, or_ = ours.subdivide(); tl, tr = theirs.subdivide()
    assert np.allclose(ol.nodes, tl.nodes, atol=2e-14)
    assert np.allclose(or_.nodes, tr.nodes, atol=2e-14)
    assert np.allclose(ours.specialize(.13, .83).nodes, theirs.specialize(.13, .83).nodes, atol=2e-14)
    assert np.allclose(ours.elevate().nodes, theirs.elevate().nodes, atol=2e-14)

def test_subdivide_simd_tail_matches_upstream():
    rng = np.random.default_rng(42)
    for dimension, degree in ((1, 12), (2, 12), (3, 7), (5, 8)):
        nodes = np.asfortranarray(rng.normal(size=(dimension, degree + 1)))
        ours, theirs = mb.Curve.from_nodes(nodes), bezier.Curve.from_nodes(nodes)
        ol, or_ = ours.subdivide(); tl, tr = theirs.subdivide()
        assert np.allclose(ol.nodes, tl.nodes, rtol=0.0, atol=2e-14)
        assert np.allclose(or_.nodes, tr.nodes, rtol=0.0, atol=2e-14)

def test_reduce_locate_and_length_match_upstream():
    nodes = np.asfortranarray([[0., 1.], [0., 2.]])
    ours, theirs = mb.Curve.from_nodes(nodes).elevate(), bezier.Curve.from_nodes(nodes).elevate()
    assert np.allclose(ours.reduce_().nodes, theirs.reduce_().nodes)
    point = np.asfortranarray([[.3], [.6]])
    assert ours.locate(point) == pytest.approx(theirs.locate(point), abs=1e-10)
    nodes = CURVES[0]
    assert mb.Curve.from_nodes(nodes).length == pytest.approx(bezier.Curve.from_nodes(nodes).length, rel=2e-6)

def test_line_and_cubic_intersections_match_upstream():
    a = np.asfortranarray([[0., .25, .75, 1.], [0., 1., -1., 0.]])
    b = np.asfortranarray([[0., 1.], [.2, .2]])
    ours = mb.Curve.from_nodes(a).intersect(mb.Curve.from_nodes(b))
    theirs = bezier.Curve.from_nodes(a).intersect(bezier.Curve.from_nodes(b))
    assert ours.shape == theirs.shape == (2, 2)
    assert np.allclose(ours, theirs, atol=2e-7)

def test_no_intersection_and_self_intersection_match_upstream():
    a = np.asfortranarray([[0., 1.], [0., 0.]])
    b = np.asfortranarray([[0., 1.], [1., 1.]])
    assert mb.Curve.from_nodes(a).intersect(mb.Curve.from_nodes(b)).shape == (2, 0)
    # This cubic has B(0.25) == B(0.75), an interior transversal crossing.
    loop = np.asfortranarray([[0., 1., 0., 9. / 13.], [0., 0., 1., -9. / 13.]])
    ours, theirs = mb.Curve.from_nodes(loop).self_intersections(), bezier.Curve.from_nodes(loop).self_intersections()
    assert np.allclose(ours, theirs, atol=2e-7)


def test_ffi_inputs_are_float64_finite_and_layout_safe():
    nodes = np.asfortranarray([[0., 1.], [0., 1.]])
    with pytest.raises(TypeError, match="float64"):
        mb.Curve.from_nodes(nodes.astype(np.float32))
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        mb.Curve(nodes.copy(order="C"), 1, copy=False)
    with pytest.raises(ValueError, match="finite"):
        mb.Curve.from_nodes(np.asfortranarray([[0., np.nan], [0., 1.]]))
    curve = mb.Curve.from_nodes(nodes, copy=False)
    assert curve.nodes is nodes
    with pytest.raises(TypeError, match="float64"):
        curve.evaluate_multi(np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="finite"):
        curve.evaluate_multi(np.array([0., np.inf]))
