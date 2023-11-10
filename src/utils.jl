"""
    PermutedMatrix{K,T} <: AbstractMatrix{T}

Structured used to reprensent the permutation of a matrix-like object. The
original matrix is stored in the `data::K` field, and the permutations are
stored in `rowperm` and `colperm`.
"""
struct PermutedMatrix{K,T} <: AbstractMatrix{T}
    data::K # original matrix
    rowperm::Vector{Int}
    colperm::Vector{Int}
    function PermutedMatrix(orig, rowperm, colperm)
        K = typeof(orig)
        T = eltype(orig)
        return new{K,T}(orig, rowperm, colperm)
    end
end
Base.size(M::PermutedMatrix) = size(M.data)

function Base.getindex(M::PermutedMatrix, i, j)
    ip = M.rowperm[i]
    jp = M.colperm[j]
    return M.data[ip, jp]
end

"""
    hilbert_cartesian_to_linear(n,x,y)

Convert the cartesian indices `x,y` into a linear index `d` using a hilbert
curve of order `n`. The coordinates `x,y` range from `0` to `n-1`, and the
output `d` ranges from `0` to `n^2-1`.

See [https://en.wikipedia.org/wiki/Hilbert_curve](https://en.wikipedia.org/wiki/Hilbert_curve).
"""
function hilbert_cartesian_to_linear(n::Integer, x, y)
    @assert ispow2(n)
    @assert 0 ≤ x ≤ n - 1
    @assert 0 ≤ y ≤ n - 1
    d = 0
    s = n >> 1
    while s > 0
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s^2 * ((3 * rx) ⊻ ry)
        x, y = _rot(n, x, y, rx, ry)
        s = s >> 1
    end
    @assert 0 ≤ d ≤ n^2 - 1
    return d
end

"""
    hilbert_linear_to_cartesian(n,d)

Convert the linear index `0 ≤ d ≤ n^2-1` into the cartesian coordinates `0 ≤ x <
n-1` and `0 ≤ y ≤ n-1` on the Hilbert curve of order `n`.

See [https://en.wikipedia.org/wiki/Hilbert_curve](https://en.wikipedia.org/wiki/Hilbert_curve).
"""
function hilbert_linear_to_cartesian(n::Integer, d)
    @assert ispow2(n)
    @assert 0 ≤ d ≤ n^2 - 1
    x, y = 0, 0
    s = 1
    while s < n
        rx = 1 & (d >> 1)
        ry = 1 & (d ⊻ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        d = d >> 2
        s = s << 1
    end
    @assert 0 ≤ x ≤ n - 1
    @assert 0 ≤ y ≤ n - 1
    return x, y
end

# auxiliary function using in hilbert curve. Rotates the points x,y
function _rot(n, x, y, rx, ry)
    if ry == 0
        if rx == 1
            x = n - 1 - x
            y = n - 1 - y
        end
        x, y = y, x
    end
    return x, y
end

function hilbert_points(n::Integer)
    @assert ispow2(n)
    xx = Int[]
    yy = Int[]
    for d in 0:(n^2-1)
        x, y = hilbert_linear_to_cartesian(n, d)
        push!(xx, x)
        push!(yy, y)
    end
    return xx, yy
end

"""
    filter_tree(f,tree,isterminal=true)

Return a vector containing all the nodes of `tree` such that
`f(node)==true`.  The argument `isterminal` can be used to control whether
to continue the search on `children` of nodes for which `f(node)==true`.
"""
function filter_tree(f, tree, isterminal = true)
    nodes = Vector{typeof(tree)}()
    return filter_tree!(f, nodes, tree, isterminal)
end

"""
    filter_tree!(filter,nodes,tree,[isterminal=true])

Like [`filter_tree`](@ref), but appends results to `nodes`.
"""
function filter_tree!(f, nodes, tree, isterminal = true)
    if f(tree)
        push!(nodes, tree)
        # terminate the search along this path if terminal=true
        isterminal || map(x -> filter_tree!(f, nodes, x, isterminal), children(tree))
    else
        # continue on on children
        map(x -> filter_tree!(f, nodes, x, isterminal), children(tree))
    end
    return nodes
end

"""
    depth(tree,acc=0)

Recursive function to compute the depth of `node` in a a tree-like structure.

Overload this function if your structure has a more efficient way to compute
`depth` (e.g. if it stores it in a field).
"""
function depth(tree, acc = 0)
    if isroot(tree)
        return acc
    else
        depth(parent(tree), acc + 1)
    end
end

"""
    build_sequence_partition(seq,nq,cost,nmax)

Partition the sequence `seq` into `nq` contiguous subsequences with a maximum of
cost of `nmax` per set. Note that if `nmax` is too small, this may not be
possible (see [`has_partition`](@ref)).
"""
function build_sequence_partition(seq, np, cost, cmax)
    acc = 0
    partition = [empty(seq) for _ in 1:np]
    k = 1
    for el in seq
        c = cost(el)
        acc += c
        if acc > cmax
            k += 1
            push!(partition[k], el)
            acc = c
            @assert k <= np "unable to build sequence partition. Value of  `cmax` may be too small."
        else
            push!(partition[k], el)
        end
    end
    return partition
end

"""
    find_optimal_cost(seq,nq,cost,tol)

Find an approximation to the cost of an optimal partitioning of `seq` into `nq`
contiguous segments. The optimal cost is the smallest number `cmax` such that
`has_partition(seq,nq,cost,cmax)` returns `true`.
"""
function find_optimal_cost(seq, np, cost = identity, tol = 1)
    lbound = Float64(maximum(cost, seq))
    ubound = Float64(sum(cost, seq))
    guess = (lbound + ubound) / 2
    while ubound - lbound ≥ tol
        if has_partition(seq, np, guess, cost)
            ubound = guess
        else
            lbound = guess
        end
        guess = (lbound + ubound) / 2
    end
    return ubound
end

"""
    find_optimal_partition(seq,nq,cost,tol=1)

Find an approximation to the optimal partition `seq` into `nq` contiguous
segments according to the `cost` function. The optimal partition is the one
which minimizes the maximum `cost` over all possible partitions of `seq` into
`nq` segments.

The generated partition is optimal up to a tolerance `tol`; for integer valued
`cost`, setting `tol=1` means the partition is optimal.
"""
function find_optimal_partition(seq, np, cost = (x) -> 1, tol = 1)
    cmax = find_optimal_cost(seq, np, cost, tol)
    p = build_sequence_partition(seq, np, cost, cmax)
    return p
end

"""
    has_partition(v,np,cost,cmax)

Given a vector `v`, determine whether or not a partition into `np` segments is
possible where the `cost` of each partition does not exceed `cmax`.
"""
function has_partition(seq, np, cmax, cost = identity)
    acc = 0
    k = 1
    for el in seq
        c = cost(el)
        acc += c
        if acc > cmax
            k += 1
            acc = c
            k > np && (return false)
        end
    end
    return true
end

"""
    struct VectorOfVectors{T}

A simple structure which behaves as a `Vector{Vector{T}}` but stores the entries
in a contiguous `data::Vector{T}` field. All vectors in the `VectorOfVectors`
are assumed to be of size `m`, and there are `k` of them, meaning this structure
can be used to represent a `m × k` matrix.

Similar to a vector-of-vectors, calling `A[i]` returns a view to the `i`-th
column.

See also: [`newcol!`](@ref)
"""
mutable struct VectorOfVectors{T}
    const data::Vector{T}
    m::Int
    k::Int
end

VectorOfVectors(T, m = 0, k = 0) = VectorOfVectors{T}(Vector{T}(undef, m * k), m, k)

"""
    newcol!(A::VectorOfVectors)

Append a new (unitialized) column to `A`, and return a view of it.
"""
function newcol!(A::VectorOfVectors)
    m, k = A.m, A.k
    is = m * k + 1
    ie = m * (k + 1)
    if ie > length(A.data)
        resize!(A.data, ie)
    end
    A.k += 1
    return view(A.data, is:ie)
end

"""
    reset!(A::VectorOfVectors)

Set the number of columns of `A` to zero, and the number of rows to zero, but
does not `resize!` the underlying data vector.
"""
reset!(A::VectorOfVectors) = (A.m = 0; A.k = 0)

function Base.getindex(A::VectorOfVectors, i)
    i <= A.k || throw(BoundsError(A, i))
    return view(A.data, (i-1)*A.m+1:i*A.m)
end

function Base.Matrix(A::VectorOfVectors{T}) where {T}
    out = Matrix{T}(undef, A.m, A.k)
    return copyto!(out, 1, A.data, 1, length(out))
end

Base.length(A::VectorOfVectors) = A.k
