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
    allow_getindex(bool)

Call this function to enable/disable the `getindex` method on `AbstractHMatrix`
and `RkMatrix`.

By default, calling `getindex` on these types will error to avoid performance
pitfalls associated with linear algebra methods falling back to a generic
implementation which uses the `getindex` method.
"""
allow_getindex(bool) = (ALLOW_GETINDEX[] = bool)

const GET_INDEX_ERROR_MSG = """
method `getindex(::Union{AbstractHMatrix,RkMatrix},args...)` has been disabled
to avoid performance pitfalls. Unless you made an explicit call to `getindex`,
this error usually means that a linear algebra routine involving an
`AbstractHMatrix` has fallen back to a generic implementation. Calling
`allow_getindex(true)` fixes this error.
"""

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
    partition_by_depth(tree)

Given a `tree`, return a `partition` vector whose `i`-th entry stores all the nodes in
`tree` with `depth=i-1`. Empty nodes are not added to the partition.
"""
function partition_by_depth(tree)
    T = typeof(tree)
    partition = Vector{Vector{T}}()
    depth = 0
    return _partition_by_depth!(partition, tree, depth)
end

function _partition_by_depth!(partition, tree, depth)
    T = typeof(tree)
    if length(partition) < depth + 1
        push!(partition, [])
    end
    length(tree) > 0 && push!(partition[depth+1], tree)
    for chd in children(tree)
        _partition_by_depth!(partition, chd, depth + 1)
    end
    return partition
end

"""
    struct VectorBuffer{T}

A buffer for vectors of type `T`. You can ask for a new vector of size `n` with
[`newbuffer(n)`](@ref). Use [`resetbuffer!`](@ref) to reset the buffer counter.

Note that freeing the buffer is done automatically by the garbage collector when
the buffer and all the vectors returned by `newbuffer` go out of scope.
"""
struct VectorBuffer{T}
    data::Vector{T}
    counter::Ref{Int}
end

VectorBuffer(T, n) = VectorBuffer{T}(Vector{T}(undef, n), 0)

function newbuffer!(buf::VectorBuffer{T}, n) where {T}
    is = buf.counter[] + 1
    ie = buf.counter[] + n
    if ie > length(buf.data)
        resize!(buf.data, 2 * length(buf.data))
    end
    buf.counter[] = ie
    return view(buf.data, is, ie)
end

resetbuffer!(buf::VectorBuffer) = (buf.counter[] = 0)

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
