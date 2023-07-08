#=
Utilities  related to partitioning the leaves of an `HMatrix` to perform `gemv`
operations
=#

"""
    struct Partition{T<:HMatrix}

A partition of the leaves of an `HMatrix`. Used to perform threaded hierarchical
multiplication.
"""
struct Partition{T<:HMatrix}
    partition::Vector{Vector{T}}
    tag::Symbol
end

"""
    const CACHED_PARTITIONS

A `WeakKeyDict` mapping a hierarhical matrix to a [`Partition`](@ref) of itself.
Used when computing e.g. the forward map (i.e. `mul!`) to avoid having to
recompute the partition for each matrix/vector product.
"""
const CACHED_PARTITIONS = WeakKeyDict{HMatrix,Partition}()

Base.hash(A::AbstractHMatrix, h::UInt) = hash(objectid(A), h)
Base.:(==)(A::AbstractHMatrix, B::AbstractHMatrix) = A === B

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
function find_optimal_cost(seq, np, cost=identity, tol=1)
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
function find_optimal_partition(seq, np, cost=(x) -> 1, tol=1)
    cmax = find_optimal_cost(seq, np, cost, tol)
    p = build_sequence_partition(seq, np, cost, cmax)
    return p
end

"""
    has_partition(v,np,cost,cmax)

Given a vector `v`, determine whether or not a partition into `np` segments is
possible where the `cost` of each partition does not exceed `cmax`.
"""
function has_partition(seq, np, cmax, cost=identity)
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
    hilbert_partition(H::HMatrix,np,cost)

Partiotion the leaves of `H` into `np` sequences of approximate equal cost (as
determined by the `cost` function) while also trying to maximize the locality of
each partition.
"""
function hilbert_partition(H::HMatrix, np=Threads.nthreads(), cost=_cost_gemv)
    # the hilbert curve will be indexed from (0,0) × (N-1,N-1), so set N to be
    # the smallest power of two larger than max(m,n), where m,n = size(H)
    m, n = size(H)
    N = max(m, n)
    N = nextpow(2, N)
    # sort the leaves by their hilbert index
    leaves = collect(AbstractTrees.Leaves(H))
    hilbert_indices = map(leaves) do leaf
        # use the center of the leaf as a cartesian index
        i, j = pivot(leaf) .- 1 .+ size(leaf) .÷ 2
        return hilbert_cartesian_to_linear(N, i, j)
    end
    p = sortperm(hilbert_indices)
    permute!(leaves, p)
    # now compute a quasi-optimal partition of leaves based `cost_mv`
    cmax = find_optimal_cost(leaves, np, cost, 1)
    _partition = build_sequence_partition(leaves, np, cost, cmax)
    partition = Partition(_partition, :hilbert)
    push!(CACHED_PARTITIONS, H => partition)
    return partition
end

# TODO: benchmark the different partitioning strategies for gemv. Is the hilber
# partition really faster than the simpler alternatives (row partition, col partition)?
function row_partition(H::HMatrix, np=Threads.nthreads(), cost=_cost_gemv)
    # sort the leaves by their row index
    leaves = filter_tree(x -> isleaf(x), H)
    row_indices = map(leaves) do leaf
        # use the center of the leaf as a cartesian index
        i, j = pivot(leaf)
        return i
    end
    p = sortperm(row_indices)
    permute!(leaves, p)
    # now compute a quasi-optimal partition of leaves based `cost_mv`
    cmax = find_optimal_cost(leaves, np, cost, 1)
    _partition = build_sequence_partition(leaves, np, cost, cmax)
    partition = Partition(_partition, :row)
    push!(CACHED_PARTITIONS, H => partition)
    return partition
end

function col_partition(H::HMatrix, np=Threads.nthreads(), cost=_cost_gemv)
    # sort the leaves by their row index
    leaves = filter_tree(x -> isleaf(x), H)
    row_indices = map(leaves) do leaf
        # use the center of the leaf as a cartesian index
        i, j = pivot(leaf)
        return j
    end
    p = sortperm(row_indices)
    permute!(leaves, p)
    # now compute a quasi-optimal partition of leaves based `cost_mv`
    cmax = find_optimal_cost(leaves, np, cost, 1)
    _partition = build_sequence_partition(leaves, np, cost, cmax)
    partition = Partition(_partition, :col)
    push!(CACHED_PARTITIONS, H => partition)
    return partition
end
