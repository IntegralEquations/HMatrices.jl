"""
    abstract type AbstractHMatrix{T} <: AbstractMatrix{T}

Abstract type for hierarchical matrices.
"""
abstract type AbstractHMatrix{T} <: AbstractMatrix{T} end

function Base.getindex(::AbstractHMatrix, args...)
    msg = """method `getindex(::AbstractHMatrix,args...)` has been disabled to
    avoid performance pitfalls. Unless you made an explicit call to `getindex`,
    this error usually means that a linear algebra routine involving an
    `AbstractHMatrix` has fallen back to a generic implementation."""
    return error(msg)
end

"""
    struct Partition{T}

A partition of the leaves of an `HMatrix`. Used to perform threaded hierarchical
multiplication.
"""
struct Partition{T}
    root::T
    partition::Vector{Vector{T}}
    tag::Symbol
end

"""
    mutable struct HMatrix{R,T} <: AbstractHMatrix{T}

A hierarchial matrix constructed from a `rowtree` and `coltree` of type `R` and
holding elements of type `T`.
"""
mutable struct HMatrix{R,T} <: AbstractHMatrix{T}
    rowtree::R
    coltree::R
    admissible::Bool
    data::Union{Matrix{T},RkMatrix{T},Nothing}
    children::Matrix{HMatrix{R,T}}
    parent::HMatrix{R,T}
    partition::Union{Nothing,Partition{HMatrix{R,T}}}
    # inner constructor which handles `nothing` fields.
    function HMatrix{R,T}(rowtree, coltree, adm, data, children, parent) where {R,T}
        if data !== nothing
            @assert (length(rowtree), length(coltree)) === size(data) "$(length(rowtree)),$(length(coltree)) != $(size(data))"
        end
        hmat = new{R,T}(rowtree, coltree, adm, data)
        hmat.children = isnothing(children) ? Matrix{HMatrix{R,T}}(undef, 0, 0) : children
        hmat.parent = isnothing(parent) ? hmat : parent
        hmat.partition = nothing
        return hmat
    end
end

# setters and getters (defined for AbstractHMatrix)
isadmissible(H::AbstractHMatrix) = H.admissible
hasdata(H::AbstractHMatrix) = !isnothing(H.data)
data(H::AbstractHMatrix) = H.data
setdata!(H::AbstractHMatrix, d) = setfield!(H, :data, d)
rowtree(H::AbstractHMatrix) = H.rowtree
coltree(H::AbstractHMatrix) = H.coltree

cluster_type(::HMatrix{R,T}) where {R,T} = R

# getcol for regular matrices
function getcol!(col, M::Matrix, j)
    @assert length(col) == size(M, 1)
    return copyto!(col, view(M, :, j))
end
function getcol!(col, adjM::Adjoint{<:Any,<:Matrix}, j)
    @assert length(col) == size(adjM, 1)
    return copyto!(col, view(adjM, :, j))
end

getcol(M::Matrix, j) = M[:, j]
getcol(adjM::Adjoint{<:Any,<:Matrix}, j) = adjM[:, j]

function getcol(H::HMatrix, j::Int)
    m, n = size(H)
    T = eltype(H)
    col = zeros(T, m)
    return getcol!(col, H, j)
end

function getcol!(col, H::HMatrix, j::Int)
    (j ∈ colrange(H)) || throw(BoundsError())
    piv = pivot(H)
    return _getcol!(col, H, j, piv)
end

function _getcol!(col, H::HMatrix, j, piv)
    if hasdata(H)
        shift = pivot(H) .- 1
        jl = j - shift[2]
        irange = rowrange(H) .- (piv[1] - 1)
        getcol!(view(col, irange), data(H), jl)
    end
    for child in children(H)
        if j ∈ colrange(child)
            _getcol!(col, child, j, piv)
        end
    end
    return col
end

function getcol(adjH::Adjoint{<:Any,<:HMatrix}, j::Int)
    # (j ∈ colrange(adjH)) || throw(BoundsError())
    m, n = size(adjH)
    T = eltype(adjH)
    col = zeros(T, m)
    return getcol!(col, adjH, j)
end

function getcol!(col, adjH::Adjoint{<:Any,<:HMatrix}, j::Int)
    piv = pivot(adjH)
    return _getcol!(col, adjH, j, piv)
end

function _getcol!(col, adjH::Adjoint{<:Any,<:HMatrix}, j, piv)
    if hasdata(adjH)
        shift = pivot(adjH) .- 1
        jl = j - shift[2]
        irange = rowrange(adjH) .- (piv[1] - 1)
        getcol!(view(col, irange), data(adjH), jl)
    end
    for child in children(adjH)
        if j ∈ colrange(child)
            _getcol!(col, child, j, piv)
        end
    end
    return col
end

# Trees interface
children(H::AbstractHMatrix) = H.children
children(H::AbstractHMatrix, idxs...) = H.children[idxs]
Base.parent(H::AbstractHMatrix) = H.parent
isleaf(H::AbstractHMatrix) = isempty(children(H))
isroot(H::AbstractHMatrix) = parent(H) === H

# interface to AbstractTrees. No children is determined by an empty tuple for
# AbstractTrees.
AbstractTrees.children(t::AbstractHMatrix) = isleaf(t) ? () : t.children
AbstractTrees.nodetype(t::AbstractHMatrix) = typeof(t)

rowrange(H::AbstractHMatrix) = index_range(H.rowtree)
colrange(H::AbstractHMatrix) = index_range(H.coltree)
rowperm(H::AbstractHMatrix) = loc2glob(rowtree(H))
colperm(H::AbstractHMatrix) = loc2glob(coltree(H))
pivot(H::AbstractHMatrix) = (rowrange(H).start, colrange(H).start)
offset(H::AbstractHMatrix) = pivot(H) .- 1

# Base.axes(H::HMatrix) = rowrange(H),colrange(H)
Base.size(H::AbstractHMatrix) = length(rowrange(H)), length(colrange(H))

function blocksize(H::AbstractHMatrix)
    return size(children(H))
end

"""
    compression_ratio(H::HMatrix)

The ratio of the uncompressed size of `H` to its compressed size.
"""
function compression_ratio(H::HMatrix)
    ns = 0 # stored entries
    nr = length(H) # represented entries
    for block in AbstractTrees.Leaves(H)
        data = block.data
        ns += num_stored_elements(data)
    end
    return nr / ns
end

num_stored_elements(M::Matrix) = length(M)

function Base.show(io::IO, hmat::HMatrix)
    isclean(hmat) || return print(io, "Dirty HMatrix")
    print(io, "HMatrix of $(eltype(hmat)) with range $(rowrange(hmat)) × $(colrange(hmat))")
    _show(io, hmat)
    return io
end
Base.show(io::IO, ::MIME"text/plain", hmat::HMatrix) = show(io, hmat)

function _show(io, hmat)
    nodes = collect(AbstractTrees.PreOrderDFS(hmat))
    @printf io "\n\t number of nodes in tree: %i" length(nodes)
    leaves = collect(AbstractTrees.Leaves(hmat))
    sparse_leaves = filter(isadmissible, leaves)
    dense_leaves = filter(!isadmissible, leaves)
    @printf(
        io,
        "\n\t number of leaves: %i (%i admissible + %i full)",
        length(leaves),
        length(sparse_leaves),
        length(dense_leaves)
    )
    rmin, rmax =
        isempty(sparse_leaves) ? (-1, -1) : extrema(x -> rank(x.data), sparse_leaves)
    @printf(io, "\n\t min rank of sparse blocks : %i", rmin)
    @printf(io, "\n\t max rank of sparse blocks : %i", rmax)
    dense_min, dense_max =
        isempty(dense_leaves) ? (-1, -1) : extrema(x -> length(x.data), dense_leaves)
    @printf(io, "\n\t min length of dense blocks : %i", dense_min)
    @printf(io, "\n\t max length of dense blocks : %i", dense_max)
    points_per_leaf = map(length, leaves)
    @printf(io, "\n\t min number of elements per leaf: %i", minimum(points_per_leaf))
    @printf(io, "\n\t max number of elements per leaf: %i", maximum(points_per_leaf))
    depth_per_leaf = map(depth, leaves)
    @printf(io, "\n\t depth of tree: %i", maximum(depth_per_leaf))
    @printf(io, "\n\t compression ratio: %f\n", compression_ratio(hmat))
    return io
end

"""
    Matrix(H::HMatrix;global_index=true)

Convert `H` to a `Matrix`. If `global_index=true` (the default), the entries are
given in the global indexing system (see [`HMatrix`](@ref) for more
information); otherwise the *local* indexing system induced by the row and
columns trees are used.
"""
Base.Matrix(hmat::HMatrix; global_index = true) = Matrix{eltype(hmat)}(hmat; global_index)
function Base.Matrix{T}(hmat::HMatrix; global_index) where {T}
    M = zeros(T, size(hmat)...)
    piv = pivot(hmat)
    for block in AbstractTrees.PreOrderDFS(hmat)
        hasdata(block) || continue
        irange = rowrange(block) .- piv[1] .+ 1
        jrange = colrange(block) .- piv[2] .+ 1
        M[irange, jrange] += Matrix(block.data)
    end
    if global_index
        P = PermutedMatrix(M, invperm(rowperm(hmat)), invperm(colperm(hmat)))
        return Matrix(P)
    else
        return M
    end
end

# deprecate assemble_hmat
@deprecate assemble_hmat assemble_hmatrix

"""
    assemble_hmatrix([T,], K,rowtree,coltree;adm=StrongAdmissibilityStd(),comp=PartialACA(),threads=true,distributed=false,global_index=true)
    assemble_hmatrix([T,], K::KernelMatrix;threads=true,distributed=false,global_index=true,[rtol],[atol],[rank])

Main routine for assembling a hierarchical matrix. The argument `K` represents
the matrix to be approximated, `rowtree` and `coltree` are tree structure
partitioning the row and column indices, respectively, `adm` can be called on a
node of `rowtree` and a node of `coltree` to determine if the block is
compressible, and `comp` is a function/functor which can compress admissible
blocks.

It is assumed that `K` supports `getindex(K,i,j)`, and that comp can be called
as `comp(K,irange::UnitRange,jrange::UnitRange)` to produce a compressed version
of `K[irange,jrange]` in the form of an [`RkMatrix`](@ref).

The type paramter `T` is used to specify the type of the entries of the matrix,
by default is inferred from `K` using `eltype(K)`.
"""
function assemble_hmatrix(
    ::Type{T},
    K,
    rowtree,
    coltree;
    adm = StrongAdmissibilityStd(3),
    comp = PartialACA(),
    global_index = use_global_index(),
    threads = use_threads(),
    distributed = false,
) where {T}
    if distributed
        _assemble_hmat_distributed(K, rowtree, coltree; adm, comp, global_index, threads)
    else
        # create first the structure. No parellelism used as this should be light.
        hmat = HMatrix{T}(rowtree, coltree, adm)
        # if needed permute kernel entries into indexing induced by trees
        global_index && (K = PermutedMatrix(K, loc2glob(rowtree), loc2glob(coltree)))
        # now assemble the data in the blocks
        if threads
            bufs = [
                (VectorOfVectors(T, 0), VectorOfVectors(T, 0)) for _ in 1:Threads.nthreads()
            ]
            _assemble_threads!(hmat, K, comp, bufs)
        else
            bufs = (VectorOfVectors(T, 0), VectorOfVectors(T, 0))
            _assemble_cpu!(hmat, K, comp, bufs)
        end
    end
end

function assemble_hmatrix(K::AbstractMatrix, args...; kwargs...)
    return assemble_hmatrix(eltype(K), K, args...; kwargs...)
end

function assemble_hmatrix(
    K::AbstractKernelMatrix;
    atol = 0,
    rank = typemax(Int),
    rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
    kwargs...,
)
    comp = PartialACA(; rtol, atol, rank)
    adm = StrongAdmissibilityStd()
    X = rowelements(K)
    Y = colelements(K)
    Xclt = ClusterTree(X)
    Yclt = ClusterTree(Y)
    return assemble_hmatrix(K, Xclt, Yclt; adm, comp, kwargs...)
end

"""
    HMatrix{T}(rowtree,coltree,adm)

Construct an empty `HMatrix` with `rowtree` and `coltree` using the
admissibility condition `adm`. This function builds the skeleton for the
hierarchical matrix, but **does not compute `data`** field in the blocks. See
[`assemble_hmatrix`](@ref) for assembling a hierarhical matrix.
"""
function HMatrix{T}(rowtree::R, coltree::R, adm) where {R,T}
    #build root
    root = HMatrix{R,T}(rowtree, coltree, false, nothing, nothing, nothing)
    # recurse
    _build_block_structure!(adm, root)
    return root
end

"""
    _build_block_structure!(adm_fun,current_node)

Recursive constructor for [`HMatrix`](@ref) block structure. Should not be called directly.
"""
function _build_block_structure!(adm, current_node::HMatrix{R,T}) where {R,T}
    X = current_node.rowtree
    Y = current_node.coltree
    if (isleaf(X) || isleaf(Y))
        current_node.admissible = false
    elseif adm(X, Y)
        current_node.admissible = true
    else
        current_node.admissible = false
        row_children = X.children
        col_children = Y.children
        children = [
            HMatrix{R,T}(r, c, false, nothing, nothing, current_node) for
            r in row_children, c in col_children
        ]
        current_node.children = children
        for child in children
            _build_block_structure!(adm, child)
        end
    end
    return current_node
end

"""
    _assemble_cpu!(hmat::HMatrix,K,comp)

Assemble data on the leaves of `hmat`. The admissible leaves are compressed
using the compressor `comp`. This function assumes the structure of `hmat` has
already been intialized, and therefore should not be called directly. See
[`HMatrix`](@ref) information on constructors.
"""
function _assemble_cpu!(hmat, K, comp, bufs)
    if isleaf(hmat) # base case
        if isadmissible(hmat)
            _assemble_sparse_block!(hmat, K, comp, bufs)
        else
            _assemble_dense_block!(hmat, K)
        end
    else
        # recurse on children
        for child in hmat.children
            _assemble_cpu!(child, K, comp, bufs)
        end
    end
    return hmat
end

"""
    _assemble_threads!(hmat::HMatrix,K,comp)

Like [`_assemble_cpu!`](@ref), but uses threads to assemble the leaves. Note
that the threads are spanwned using `Threads.@spawn`, which means they are
spawned on the same worker as the caller.
"""
function _assemble_threads!(hmat, K, comp, bufs)
    # FIXME: ideally something like `omp for schedule(guided)` should be used here
    # to avoid spawning too many (small) tasks. In the absece of such scheduling
    # strategy in julia at the moment (v1.6), we resort to manually limiting the size
    # of the tasks by directly calling the serial method for blocks which are
    # smaller than a given length (1000^2 here).
    blocks = filter_tree(hmat, true) do x
        return (isleaf(x) || length(x) < 1000 * 1000)
    end
    sort!(blocks; lt = (x, y) -> length(x) < length(y), rev = true)
    n = length(blocks)
    @sync for i in 1:n
        Threads.@spawn begin
            tid = Threads.threadid()
            _assemble_cpu!(blocks[i], K, comp, bufs[tid])
            tid == Threads.threadid() || (@warn "thread id changed!")
        end
    end
    return hmat
end

function _assemble_sparse_block!(hmat, K, comp, bufs)
    return hmat.data = comp(K, hmat.rowtree, hmat.coltree, bufs)
end

function _assemble_dense_block!(hmat, K)
    irange = rowrange(hmat)
    jrange = colrange(hmat)
    T = eltype(hmat)
    out = Matrix{T}(undef, length(irange), length(jrange))
    getblock!(out, K, irange, jrange)
    hmat.data = out
    return hmat
end

hasdata(adjH::Adjoint{<:Any,<:HMatrix}) = hasdata(adjH.parent)
data(adjH::Adjoint{<:Any,<:HMatrix}) = adjoint(data(adjH.parent))
children(adjH::Adjoint{<:Any,<:HMatrix}) = adjoint(children(adjH.parent))
pivot(adjH::Adjoint{<:Any,<:HMatrix}) = reverse(pivot(adjH.parent))
offset(adjH::Adjoint{<:Any,<:HMatrix}) = pivot(adjH) .- 1
rowrange(adjH::Adjoint{<:Any,<:HMatrix}) = colrange(adjH.parent)
colrange(adjH::Adjoint{<:Any,<:HMatrix}) = rowrange(adjH.parent)
isleaf(adjH::Adjoint{<:Any,<:HMatrix}) = isleaf(adjH.parent)

Base.size(adjH::Adjoint{<:Any,<:HMatrix}) = reverse(size(adjH.parent))

function Base.show(io::IO, adjH::Adjoint{<:Any,<:HMatrix})
    hmat = parent(adjH)
    isclean(hmat) || return print(io, "Dirty HMatrix")
    print(
        io,
        "Adjoint HMatrix of $(eltype(hmat)) with range $(rowrange(adjH)) × $(colrange(adjH))",
    )
    _show(io, hmat)
    return io
end
Base.show(io::IO, ::MIME"text/plain", adjH::Adjoint{<:Any,<:HMatrix}) = show(io, adjH)

"""
    struct StrongAdmissibilityStd

Two blocks are admissible under this condition if the minimum of their
`diameter` is smaller than `eta` times the `distance` between them, where
`eta::Float64` is a parameter.

## Usage:
```julia
adm = StrongAdmissibilityStd(;eta=2.0)
adm(Xnode,Ynode)
```
"""
Base.@kwdef struct StrongAdmissibilityStd
    eta::Float64 = 3.0
end

function (adm::StrongAdmissibilityStd)(left_node, right_node)
    diam_min = minimum(diameter, (left_node, right_node))
    dist = distance(left_node, right_node)
    return diam_min < adm.eta * dist
end

"""
    struct WeakAdmissibilityStd

Two blocks are admissible under this condition if the `distance`
between them is positive.
"""
struct WeakAdmissibilityStd end

(adm::WeakAdmissibilityStd)(left_node, right_node) = distance(left_node, right_node) > 0

"""
    isclean(H::HMatrix)

Return `true` if all leaves of `H` have data, and if the leaves are the only
nodes containing data. This is the normal state of an ℋ-matrix, but during
intermediate stages of a computation data may be associated with non-leaf nodes
for convenience.
"""
function isclean(H::HMatrix)
    for node in AbstractTrees.PreOrderDFS(H)
        if isleaf(node)
            if !hasdata(node)
                @warn "leaf node without data found"
                return false
            end
        else
            if hasdata(node)
                @warn "data found on non-leaf node"
                return false
            end
        end
    end
    return true
end

function depth(tree::HMatrix, acc = 0)
    if isroot(tree)
        return acc
    else
        depth(parent(tree), acc + 1)
    end
end

function Base.zero(H::HMatrix)
    H0 = deepcopy(H)
    rmul!(H0, 0)
    return H0
end

function compress!(H::HMatrix, comp)
    @assert isclean(H)
    for leaf in AbstractTrees.Leaves(H)
        d = data(leaf)
        if d isa RkMatrix
            compress!(d, comp)
        end
    end
    return H
end

############################################################################################
# Recipes
############################################################################################
@recipe function f(hmat::HMatrix)
    legend --> false
    grid --> false
    aspect_ratio --> :equal
    yflip := true
    seriestype := :shape
    linecolor --> :black
    # all leaves
    for block in AbstractTrees.Leaves(hmat)
        @series begin
            if isadmissible(block)
                fillcolor --> :blue
                seriesalpha --> 1 / compression_ratio(block.data)
            else
                fillcolor --> :red
                seriesalpha --> 0.3
            end
            pt1 = pivot(block)
            pt2 = pt1 .+ size(block) .- 1
            y1, y2 = pt1[1], pt2[1]
            x1, x2 = pt1[2], pt2[2]
            # annotations := ((x1+x2)/2,(y1+y2)/2, rank(block.data))
            [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1]
        end
    end
end

"""
    hilbert_partition(H::HMatrix,np,cost)

Partiotion the leaves of `H` into `np` sequences of approximate equal cost (as
determined by the `cost` function) while also trying to maximize the locality of
each partition.
"""
function hilbert_partition(H::HMatrix, np = Threads.nthreads(), cost = _cost_gemv)
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
    return build_sequence_partition(leaves, np, cost, cmax)
end

"""
    row_partition(H::HMatrix,np,cost)

Similar to [`hilbert_partition`](@ref), but attempts to partition the leaves of
`H` by row.
"""
function row_partition(H::HMatrix, np = Threads.nthreads(), cost = _cost_gemv)
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
    return build_sequence_partition(leaves, np, cost, cmax)
end

"""
    col_partition(H::HMatrix,np,cost)

Similar to [`hilbert_partition`](@ref), but attempts to partition the leaves of
`H` by column.
"""
function col_partition(H::HMatrix, np = Threads.nthreads(), cost = _cost_gemv)
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
    return build_sequence_partition(leaves, np, cost, cmax)
end

function partition!(s::Symbol, H::HMatrix, np = Threads.nthreads(), cost = _cost_gemv)
    p = if s == :hilbert
        hilbert_partition(H, np, cost)
    elseif s == :row
        row_partition(H, np, cost)
    elseif s == :col
        col_partition(H, np, cost)
    else
        error("Unknown partitioning strategy: $s")
    end
    isnothing(H.partition) || (@warn "overwriting existing partition")
    H.partition = Partition(H, p, s)
    return H
end

# add a uniform scaling to an HMatrix return an HMatrix
function LinearAlgebra.axpy!(a, X::UniformScaling, Y::HMatrix)
    @assert isclean(Y)
    if hasdata(Y)
        d = data(Y)
        @assert d isa Matrix
        n = min(size(d)...)
        for i in 1:n
            d[i, i] += a * X.λ
        end
    else
        n = min(blocksize(Y)...)
        for i in 1:n
            axpy!(a, X, children(Y)[i, i])
        end
    end
    return Y
end

Base.:(+)(X::UniformScaling, Y::HMatrix) = axpy!(true, X, deepcopy(Y))
Base.:(+)(X::HMatrix, Y::UniformScaling) = Y + X

# adding a sparse matrix to an HMatrix is allowed if the sparse matrix is null
# on admissble blocks. This arises when adding a correction to BIE matrices.
function LinearAlgebra.axpy!(a, X::AbstractSparseArray{<:Any,<:Any,2}, Y::HMatrix)
    T = eltype(Y)
    if isleaf(Y)
        rows = rowvals(X)
        vals = nonzeros(X)
        irange = rowrange(Y)
        jrange = colrange(Y)
        for j in jrange
            for idx in nzrange(X, j)
                i = rows[idx]
                if i < irange.start
                    continue
                elseif i <= irange.stop # i ∈ irange
                    @assert !isadmissible(Y) "not possible to add sparse matrix to admissible block"
                    M = data(Y)::Matrix{T} #
                    M[i-irange.start+1, j-jrange.start+1] += a * vals[idx]
                else
                    break # go to next column
                end
            end
        end
    else # has children
        for child in children(Y)
            axpy!(a, X, child)
        end
    end
    return Y
end

Base.:(+)(X::AbstractSparseArray, Y::HMatrix) = axpy!(true, X, deepcopy(Y))
Base.:(+)(X::HMatrix, Y::AbstractSparseArray) = Y + X
