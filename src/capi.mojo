"""C ABI for hot paths of planar Bezier curves. Storage belongs to Python."""

from std.sys.info import simd_width_of as simdwidthof

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def choose(n: Int, k: Int) -> Float64:
    if k < 0 or k > n:
        return 0.0
    var r = Float64(1)
    var kk = k
    if kk > n - kk:
        kk = n - kk
    for i in range(1, kk + 1):
        r = r * Float64(n - kk + i) / Float64(i)
    return r


def evaluate_component(nodes: Ptr, dimension: Int, degree: Int, s: Float64, d: Int) -> Float64:
    if s <= 0.0:
        return nodes[d]
    if s >= 1.0:
        return nodes[degree * dimension + d]
    var one_minus = 1.0 - s
    var coefficient = 1.0
    for _ in range(degree):
        coefficient *= one_minus
    var value = nodes[d] * coefficient
    for j in range(1, degree + 1):
        coefficient = coefficient * Float64(degree - j + 1) * s / (Float64(j) * one_minus)
        value += nodes[j * dimension + d] * coefficient
    return value


def evaluate(nodes: Ptr, dimension: Int, degree: Int, s: Float64, result: Ptr):
    for d in range(dimension):
        result[d] = evaluate_component(nodes, dimension, degree, s, d)


def derivative_component(nodes: Ptr, dimension: Int, degree: Int, s: Float64, d: Int) -> Float64:
    if degree == 0:
        return 0.0
    if s <= 0.0:
        return Float64(degree) * (nodes[dimension + d] - nodes[d])
    if s >= 1.0:
        return Float64(degree) * (nodes[degree * dimension + d] - nodes[(degree - 1) * dimension + d])
    var one_minus = 1.0 - s
    var coefficient = 1.0
    for _ in range(degree - 1):
        coefficient *= one_minus
    var value = Float64(degree) * (nodes[dimension + d] - nodes[d]) * coefficient
    for j in range(1, degree):
        coefficient = coefficient * Float64(degree - j) * s / (Float64(j) * one_minus)
        value += Float64(degree) * (nodes[(j + 1) * dimension + d] - nodes[j * dimension + d]) * coefficient
    return value


def evaluate_vector(nodes: Ptr, dimension: Int, degree: Int, values: Ptr, start: Int, result: Ptr):
    var s = values.load[width=W](start)
    var one_minus = SIMD[DType.float64, W](1.0) - s
    for d in range(dimension):
        var coefficient = SIMD[DType.float64, W](1.0)
        for _ in range(degree):
            coefficient *= one_minus
        var value = SIMD[DType.float64, W](nodes[d]) * coefficient
        for k in range(1, degree + 1):
            coefficient = coefficient * SIMD[DType.float64, W](Float64(degree - k + 1)) * s / (SIMD[DType.float64, W](Float64(k)) * one_minus)
            value += SIMD[DType.float64, W](nodes[k * dimension + d]) * coefficient
        if dimension == 1:
            result.store(start, value)
        else:
            for lane in range(W):
                result[(start + lane) * dimension + d] = value[lane]
    for lane in range(W):
        if values[start + lane] <= 0.0 or values[start + lane] >= 1.0:
            evaluate(nodes, dimension, degree, values[start + lane], result + (start + lane) * dimension)


def evaluate_range(nodes: Ptr, dimension: Int, degree: Int, values: Ptr, start: Int, end: Int, result: Ptr):
    var vector_end = start + ((end - start) // W) * W
    for j in range(start, vector_end, W):
        evaluate_vector(nodes, dimension, degree, values, j, result)
    for j in range(vector_end, end):
        evaluate(nodes, dimension, degree, values[j], result + j * dimension)


@export("mbz_evaluate_multi")
def mbz_evaluate_multi(nodes_addr: Int, dimension: Int, degree: Int, values_addr: Int, count: Int, result_addr: Int) abi("C"):
    if dimension <= 0 or degree < 0 or count <= 0:
        return
    var nodes = p(nodes_addr)
    var values = p(values_addr)
    var result = p(result_addr)
    evaluate_range(nodes, dimension, degree, values, 0, count, result)


@export("mbz_subdivide")
def mbz_subdivide(nodes_addr: Int, dimension: Int, degree: Int, left_addr: Int, right_addr: Int) abi("C"):
    if dimension <= 0 or degree < 0:
        return
    var nodes = p(nodes_addr)
    var left = p(left_addr)
    var right = p(right_addr)
    var half_power = 1.0
    for i in range(degree + 1):
        for d in range(dimension):
            var lval = Float64(0)
            var rval = Float64(0)
            for j in range(i + 1):
                var weight = choose(i, j) * half_power
                lval += weight * nodes[j * dimension + d]
                rval += weight * nodes[(degree - i + j) * dimension + d]
            left[i * dimension + d] = lval
            right[(degree - i) * dimension + d] = rval
        half_power *= 0.5


@export("mbz_elevate")
def mbz_elevate(nodes_addr: Int, dimension: Int, degree: Int, result_addr: Int) abi("C"):
    if dimension <= 0 or degree < 0:
        return
    var nodes = p(nodes_addr)
    var result = p(result_addr)
    for d in range(dimension):
        result[d] = nodes[d]
        result[(degree + 1) * dimension + d] = nodes[degree * dimension + d]
    for i in range(1, degree + 1):
        var alpha = Float64(i) / Float64(degree + 1)
        for d in range(dimension):
            result[i * dimension + d] = alpha * nodes[(i - 1) * dimension + d] + (1.0 - alpha) * nodes[i * dimension + d]


@export("mbz_refine_candidates")
def mbz_refine_candidates(a_addr: Int, degree_a: Int, b_addr: Int, degree_b: Int, pairs_addr: Int, count: Int, valid_addr: Int) abi("C"):
    if degree_a < 0 or degree_b < 0 or count <= 0:
        return
    var a = p(a_addr)
    var b = p(b_addr)
    var pairs = p(pairs_addr)
    var valid = IPtr(unsafe_from_address=valid_addr)
    for i in range(count):
        var t = pairs[2 * i]
        var u = pairs[2 * i + 1]
        for _ in range(24):
            var ax = evaluate_component(a, 2, degree_a, t, 0)
            var ay = evaluate_component(a, 2, degree_a, t, 1)
            var bx = evaluate_component(b, 2, degree_b, u, 0)
            var by = evaluate_component(b, 2, degree_b, u, 1)
            var dax = derivative_component(a, 2, degree_a, t, 0)
            var day = derivative_component(a, 2, degree_a, t, 1)
            var dbx = derivative_component(b, 2, degree_b, u, 0)
            var dby = derivative_component(b, 2, degree_b, u, 1)
            var determinant = day * dbx - dax * dby
            if abs(determinant) < 1.0e-15:
                break
            var dt = ((ax - bx) * dby - (ay - by) * dbx) / determinant
            var du = (day * (ax - bx) - dax * (ay - by)) / determinant
            var nt = min(1.0, max(0.0, t + dt))
            var nu = min(1.0, max(0.0, u + du))
            if abs(nt - t) + abs(nu - u) < 1.0e-14:
                t = nt
                u = nu
                break
            t = nt
            u = nu
        pairs[2 * i] = t
        pairs[2 * i + 1] = u
        var dx = abs(evaluate_component(a, 2, degree_a, t, 0) - evaluate_component(b, 2, degree_b, u, 0))
        var dy = abs(evaluate_component(a, 2, degree_a, t, 1) - evaluate_component(b, 2, degree_b, u, 1))
        valid[i] = 1 if max(dx, dy) <= 5.0e-8 else 0


def bbox(nodes: Ptr, offset: Int, count: Int, result: Ptr):
    result[0] = nodes[offset]
    result[1] = result[0]
    result[2] = nodes[offset + 1]
    result[3] = result[2]
    for i in range(1, count):
        var x = nodes[offset + 2 * i]
        var y = nodes[offset + 2 * i + 1]
        if x < result[0]:
            result[0] = x
        if x > result[1]:
            result[1] = x
        if y < result[2]:
            result[2] = y
        if y > result[3]:
            result[3] = y


def split_curve(work: Ptr, source: Int, left: Int, right: Int, point_count: Int, other_offset: Int, other_count: Int, scratch: Int):
    for i in range(2 * point_count):
        work[scratch + i] = work[source + i]
    for d in range(2):
        work[left + d] = work[scratch + d]
        work[right + 2 * (point_count - 1) + d] = work[scratch + 2 * (point_count - 1) + d]
    for level in range(1, point_count):
        for i in range(point_count - level):
            for d in range(2):
                work[scratch + 2 * i + d] = 0.5 * (work[scratch + 2 * i + d] + work[scratch + 2 * (i + 1) + d])
        for d in range(2):
            work[left + 2 * level + d] = work[scratch + d]
            work[right + 2 * (point_count - 1 - level) + d] = work[scratch + 2 * (point_count - 1 - level) + d]
    for i in range(2 * other_count):
        work[left + other_offset + i] = work[source + other_offset + i]
        work[right + other_offset + i] = work[source + other_offset + i]


def split_second(work: Ptr, source: Int, left: Int, right: Int, first_count: Int, point_count: Int, scratch: Int):
    var second_offset = 2 * first_count
    for i in range(2 * first_count):
        work[left + i] = work[source + i]
        work[right + i] = work[source + i]
    for i in range(2 * point_count):
        work[scratch + i] = work[source + second_offset + i]
    for d in range(2):
        work[left + second_offset + d] = work[scratch + d]
        work[right + second_offset + 2 * (point_count - 1) + d] = work[scratch + 2 * (point_count - 1) + d]
    for level in range(1, point_count):
        for i in range(point_count - level):
            for d in range(2):
                work[scratch + 2 * i + d] = 0.5 * (work[scratch + 2 * i + d] + work[scratch + 2 * (i + 1) + d])
        for d in range(2):
            work[left + second_offset + 2 * level + d] = work[scratch + d]
            work[right + second_offset + 2 * (point_count - 1 - level) + d] = work[scratch + 2 * (point_count - 1 - level) + d]


@export("mbz_intersect_candidates")
def mbz_intersect_candidates(a_addr: Int, degree_a: Int, b_addr: Int, degree_b: Int, tolerance: Float64, max_depth: Int, work_addr: Int, max_frames: Int, pairs_addr: Int, max_pairs: Int) abi("C") -> Int:
    if degree_a < 0 or degree_b < 0 or tolerance < 0.0 or max_depth < 0 or max_frames < 1 or max_pairs < 1:
        return -3
    var a = p(a_addr)
    var b = p(b_addr)
    var work = p(work_addr)
    var pairs = p(pairs_addr)
    var na = degree_a + 1
    var nb = degree_b + 1
    var controls = 2 * (na + nb)
    var record = controls + 5
    for i in range(2 * na):
        work[i] = a[i]
    for i in range(2 * nb):
        work[2 * na + i] = b[i]
    work[controls] = 0.0
    work[controls + 1] = 1.0
    work[controls + 2] = 0.0
    work[controls + 3] = 1.0
    work[controls + 4] = 0.0
    var top = 1
    var written = 0
    var scratch = max_frames * record
    while top > 0:
        top -= 1
        var base = top * record
        bbox(work, base, na, work + scratch)
        bbox(work, base + 2 * na, nb, work + scratch + 4)
        var ax0 = work[scratch]; var ax1 = work[scratch + 1]
        var ay0 = work[scratch + 2]; var ay1 = work[scratch + 3]
        var bx0 = work[scratch + 4]; var bx1 = work[scratch + 5]
        var by0 = work[scratch + 6]; var by1 = work[scratch + 7]
        if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
            continue
        var aw = ax1 - ax0
        if ay1 - ay0 > aw:
            aw = ay1 - ay0
        var bw = bx1 - bx0
        if by1 - by0 > bw:
            bw = by1 - by0
        var depth = Int(work[base + controls + 4])
        if (aw <= tolerance and bw <= tolerance) or depth >= max_depth:
            if written >= max_pairs:
                return -2
            pairs[2 * written] = 0.5 * (work[base + controls] + work[base + controls + 1])
            pairs[2 * written + 1] = 0.5 * (work[base + controls + 2] + work[base + controls + 3])
            written += 1
            continue
        if top + 2 > max_frames:
            return -1
        # Reuse the popped frame for one child. The split routines first copy
        # the changing control polygon to scratch, so `child0 == base` is safe.
        var child0 = base
        var child1 = (top + 1) * record
        if aw >= bw:
            split_curve(work, base, child0, child1, na, 2 * na, nb, scratch)
            var mid = 0.5 * (work[base + controls] + work[base + controls + 1])
            work[child0 + controls] = work[base + controls]; work[child0 + controls + 1] = mid
            work[child1 + controls] = mid; work[child1 + controls + 1] = work[base + controls + 1]
            work[child0 + controls + 2] = work[base + controls + 2]; work[child0 + controls + 3] = work[base + controls + 3]
            work[child1 + controls + 2] = work[base + controls + 2]; work[child1 + controls + 3] = work[base + controls + 3]
        else:
            split_second(work, base, child0, child1, na, nb, scratch)
            var mid = 0.5 * (work[base + controls + 2] + work[base + controls + 3])
            work[child0 + controls] = work[base + controls]; work[child0 + controls + 1] = work[base + controls + 1]
            work[child1 + controls] = work[base + controls]; work[child1 + controls + 1] = work[base + controls + 1]
            work[child0 + controls + 2] = work[base + controls + 2]; work[child0 + controls + 3] = mid
            work[child1 + controls + 2] = mid; work[child1 + controls + 3] = work[base + controls + 3]
        work[child0 + controls + 4] = Float64(depth + 1)
        work[child1 + controls + 4] = Float64(depth + 1)
        top += 2
    return written
