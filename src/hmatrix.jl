abstract type AbstractHMatrix{T} <: AbstractMatrix{T} end

function Base.getindex(H::AbstractHMatrix, i::Int, j::Int)
    # NOTE: you may disable `getindex` to avoid having code that will work, but be
    # horribly slow because it falls back to some generic implementation in
    # LinearAlgebra. The downside is that the `show` method, which will usually
    # call `getindex` for `AbstractMatrix`, has to be overloaded too. One
    # options is not not inherit from `AbstractMatrix` since we don't have an
    # efficient `getindex` method in any case. The downside is that some
    # convenient functionality of `AbstractMatrix` will be lost.
    msg = """
    method `getindex(::AbstractHMatrix,args...)` has been disabled to avoid
    performance pitfalls. Unless you made an explicit call to `getindex`, this
    error usually means that a linear algebra routine involving an
    `AbstractHMatrix` has fallen back to a generic implementation.
    """
    if ALLOW_GETINDEX[]
        shift = pivot(H) .- 1
        _getindex(H, i + shift[1], j + shift[2])
    else
        error(msg)
    end
end

function _getindex(H::AbstractHMatrix, i::Int, j::Int)
    (i ∈ rowrange(H)) && (j ∈ colrange(H)) || throw(BoundsError(H, (i, j)))
    acc = zero(eltype(H))
    shift = pivot(H) .- 1
    if hasdata(H)
        il = i - shift[1]
        jl = j - shift[2]
        acc += data(H)[il, jl]
    end
    for child in children(H)
        if (i ∈ rowrange(child)) && (j ∈ colrange(child))
            acc += _getindex(child, i, j)
        end
    end
    return acc
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
    root::HMatrix{R,T}
    # inner constructor which handles `nothing` fields.
    function HMatrix{R,T}(rowtree, coltree, adm, parent=nothing, data=nothing,
                          children=nothing) where {R,T}
        if data !== nothing
            @assert (length(rowtree), length(coltree)) === size(data) "$(length(rowtree)),$(length(coltree)) != $(size(data))"
        end
        hmat = new{R,T}(rowtree, coltree, adm, data)
        hmat.children = isnothing(children) ? Matrix{HMatrix{R,T}}(undef, 0, 0) : children
        hmat.parent = isnothing(parent) ? hmat : parent
        hmat.root = isnothing(parent) ? hmat : parent.root
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
leaves(H::AbstractHMatrix) = H.leaves
root(H::AbstractHMatrix) = H.root

cluster_type(::HMatrix{R,T}) where {R,T} = R

# Trees interface
children(H::AbstractHMatrix) = H.children
children(H::AbstractHMatrix, idxs...) = H.children[idxs]
parent(H::AbstractHMatrix) = H.parent
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
    @printf(io,
            "\n\t number of leaves: %i (%i admissible + %i full)",
            length(leaves),
            length(sparse_leaves),
            length(dense_leaves))
    rmin, rmax = isempty(sparse_leaves) ? (-1, -1) :
                 extrema(x -> rank(x.data), sparse_leaves)
    @printf(io, "\n\t min rank of sparse blocks : %i", rmin)
    @printf(io, "\n\t max rank of sparse blocks : %i", rmax)
    dense_min, dense_max = isempty(dense_leaves) ? (-1, -1) :
                           extrema(x -> length(x.data), dense_leaves)
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
    Matrix(H::HMatrix;global_index=false)

Convert `H` to a `Matrix`. If `global_index=true`, the entries are given in the
global indexing system (see [`HMatrix`](@ref) for more information); otherwise
the *local* indexing system induced by the row and columns trees are used
(default).
"""
Matrix(hmat::HMatrix; global_index=true) = Matrix{eltype(hmat)}(hmat; global_index)
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

"""
    assemble_hmat(K,rowtree,coltree;adm=StrongAdmissibilityStd(),comp=PartialACA(),threads=true,distributed=false,global_index=true)
    assemble_hmat(K::KernelMatrix;threads=true,distributed=false,global_index=true,[rtol],[atol],[rank])

Main routine for assembling a hierarchical matrix. The argument `K` represents
the matrix to be approximated, `rowtree` and `coltree` are tree structure
partitioning the row and column indices, respectively, `adm` can be called on a
node of `rowtree` and a node of `coltree` to determine if the block is
compressible, and `comp` is a function/functor which can compress admissible
blocks.

It is assumed that `K` supports `getindex(K,i,j)`, and that comp can be called
as `comp(K,irange::UnitRange,jrange::UnitRange)` to produce a compressed version
of `K[irange,jrange]` in the form of an [`RkMatrix`](@ref).
"""
function assemble_hmat(K,
                       rowtree,
                       coltree;
                       adm=StrongAdmissibilityStd(3),
                       comp=PartialACA(),
                       global_index=use_global_index(),
                       threads=use_threads(),
                       distributed=false)
    T = eltype(K)
    R = typeof(rowtree)
    if distributed
        _assemble_hmat_distributed(K, rowtree, coltree; adm, comp, global_index, threads)
    else
        # if needed permute kernel entries into indexing induced by trees
        global_index && (K = PermutedMatrix(K, loc2glob(rowtree), loc2glob(coltree)))
        # create the root, then recurse
        parent = nothing
        root = HMatrix{R,T}(rowtree, coltree, adm(rowtree, coltree), parent)
        # recurse and build the blocks
        @timeit_debug "assembling hmatrix" begin
            nt = threads ? Threads.nthreads() : 1
            @info "Assembling HMatrix on $nt thread(s)"
            _assemble_hmat!(root, K, comp, adm, threads)
        end
    end
    return root
end

function assemble_hmat(K::AbstractKernelMatrix;
                       atol=0,
                       rank=typemax(Int),
                       rtol=atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
                       kwargs...)
    comp = PartialACA(; rtol, atol, rank)
    adm = StrongAdmissibilityStd(3)
    X = rowelements(K)
    Y = colelements(K)
    Xclt = ClusterTree(X)
    Yclt = ClusterTree(Y)
    return assemble_hmat(K, Xclt, Yclt; adm, comp, kwargs...)
end

"""
    _assemble_hmat!(hmat,K,comp,adm)

Recursive constructor for [`HMatrix`](@ref) called by [`assemble_hmat`](@ref) to
assemble data on the leaves of `hmat`.
"""
function _assemble_hmat!(H::HMatrix{R,T}, K, comp, adm, threads) where {R,T}
    # _add_to_indexer!(current_node)
    X = rowtree(H)
    Y = coltree(H)
    @sync begin
        if (isleaf(X) || isleaf(Y))
            H.admissible = false
            @usespawn threads _assemble_dense_block!(H, K)
        elseif adm(X, Y)
            H.admissible = true
            @usespawn threads _assemble_sparse_block!(H, K, comp)
        else
            H.admissible = false
            rows = X.children
            cols = Y.children
            children = [HMatrix{R,T}(r, c, false, H) for r in rows, c in cols]
            H.children = children
            for child in children
                @usespawn threads _assemble_hmat!(child, K, comp, adm, threads)
            end
        end
    end
    return H
end

function _assemble_sparse_block!(hmat, K, comp)
    return hmat.data = comp(K, hmat.rowtree, hmat.coltree)
end

function _assemble_dense_block!(hmat, K)
    irange = rowrange(hmat)
    jrange = colrange(hmat)
    hmat.data = K[irange, jrange]
    return hmat
end

hasdata(adjH::Adjoint{<:Any,<:HMatrix}) = hasdata(adjH.parent)
data(adjH::Adjoint{<:Any,<:HMatrix}) = hasdata(adjH) ? adjoint(data(adjH.parent)) : nothing
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
    print(io,
          "Adjoint HMatrix of $(eltype(hmat)) with range $(rowrange(adjH)) × $(colrange(adjH))")
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

function depth(tree::HMatrix, acc=0)
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

Base.getindex(H::HMatrix, ::Colon, j) = getcol(H, j)

function getcol!(col, M::Matrix, j, append::Val{T}=Val(false)) where {T}
    @assert length(col) == size(M, 1)
    if T
        axpy!(true, col, view(M, :, j))
    else
        copyto!(col, view(M, :, j))
    end
end
function getcol!(col, adjM::Adjoint{<:Any,<:Matrix}, j, append::Val{T}=Val(false)) where {T}
    @assert length(col) == size(adjM, 1)
    if T
        axpy!(true, col, view(M, :, j))
    else
        copyto!(col, view(adjM, :, j))
    end
end

getcol(M::Matrix, j) = M[:, j]
getcol(adjM::Adjoint{<:Any,<:Matrix}, j) = adjM[:, j]

function getcol!(col, H::HMatrix, j::Int, append::Val{T}=Val(false)) where {T}
    (j ∈ colrange(H)) || throw(BoundsError())
    piv = pivot(H)
    return _getcol!(col, H, j, piv, append)
end

function _getcol!(col, H::HMatrix, j, piv, append)
    if hasdata(H)
        shift = pivot(H) .- 1
        jl = j - shift[2]
        irange = rowrange(H) .- (piv[1] - 1)
        getcol!(view(col, irange), data(H), jl, append)
    end
    for child in children(H)
        if j ∈ colrange(child)
            _getcol!(col, child, j, piv, append)
        end
    end
    return col
end

Base.getindex(adjH::Adjoint{<:Any,<:HMatrix}, ::Colon, j) = getcol(adjH, j)

function getcol!(col, adjH::Adjoint{<:Any,<:HMatrix}, j::Int,
                 append::Val{T}=Val(false)) where {T}
    piv = pivot(adjH)
    return _getcol!(col, adjH, j, piv, append)
end

function _getcol!(col, adjH::Adjoint{<:Any,<:HMatrix}, j, piv, append)
    if hasdata(adjH)
        shift = pivot(adjH) .- 1
        jl = j - shift[2]
        irange = rowrange(adjH) .- (piv[1] - 1)
        getcol!(view(col, irange), data(adjH), jl, append)
    end
    for child in children(adjH)
        if j ∈ colrange(child)
            _getcol!(col, child, j, piv, append)
        end
    end
    return col
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
