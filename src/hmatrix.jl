"""
    mutable struct HMatrix{R,T} <: AbstractMatrix{T}

A hierarchial matrix constructed from a `rowtree` and `coltree` of type `R` and
holding elements of type `T`.
"""
mutable struct HMatrix{R,T} <: AbstractMatrix{T}
    rowtree::R
    coltree::R
    admissible::Bool
    data::Union{Matrix{T},RkMatrix{T},Nothing}
    children::Matrix{HMatrix{R,T}}
    parent::HMatrix{R,T}
    # inner constructor which handles `nothing` fields.
    function HMatrix{R,T}(rowtree, coltree, adm, data, children, parent) where {R,T}
        if data !== nothing
            @assert (length(rowtree), length(coltree)) === size(data) "$(length(rowtree)),$(length(coltree)) != $(size(data))"
        end
        hmat = new{R,T}(rowtree, coltree, adm, data)
        hmat.children = isnothing(children) ? Matrix{HMatrix{R,T}}(undef, 0, 0) : children
        hmat.parent = isnothing(parent) ? hmat : parent
        return hmat
    end
end

function Base.getindex(::HMatrix, args...)
    msg = """method `getindex(::HMatrix,args...)` has been disabled to
    avoid performance pitfalls. Unless you made an explicit call to `getindex`,
    this error usually means that a linear algebra routine involving an
    `HMatrix` has fallen back to a generic implementation."""
    return error(msg)
end

# setters and getters (defined for HMatrix)
isadmissible(H::HMatrix) = H.admissible
hasdata(H::HMatrix)      = !isnothing(H.data)
data(H::HMatrix)         = H.data
setdata!(H::HMatrix, d)  = setfield!(H, :data, d)
rowtree(H::HMatrix)      = H.rowtree
coltree(H::HMatrix)      = H.coltree

# getcol for regular matrices
function getcol!(col, M::Base.Matrix, j)
    @assert length(col) == size(M, 1)
    return copyto!(col, view(M, :, j))
end
function getcol!(col, adjM::Adjoint{<:Any,<:Matrix}, j)
    @assert length(col) == size(adjM, 1)
    return copyto!(col, view(adjM, :, j))
end

getcol(M::Base.Matrix, j) = M[:, j]
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
children(H::HMatrix) = H.children
children(H::HMatrix, idxs...) = H.children[idxs]
Base.parent(H::HMatrix) = H.parent
isleaf(H::HMatrix) = isempty(children(H))
isroot(H::HMatrix) = parent(H) === H

rowrange(H::HMatrix) = index_range(H.rowtree)
colrange(H::HMatrix) = index_range(H.coltree)
rowperm(H::HMatrix) = loc2glob(rowtree(H))
colperm(H::HMatrix) = loc2glob(coltree(H))
pivot(H::HMatrix) = (rowrange(H).start, colrange(H).start)
offset(H::HMatrix) = pivot(H) .- 1

# Base.axes(H::HMatrix) = rowrange(H),colrange(H)
Base.size(H::HMatrix) = length(rowrange(H)), length(colrange(H))

function blocksize(H)
    return size(children(H))
end

"""
    compression_ratio(H::HMatrix)

The ratio of the uncompressed size of `H` to its compressed size. A
`compression_ratio` of `10` means it would have taken 10 times more memory to
store `H` as a dense matrix.
"""
function compression_ratio(H::HMatrix)
    ns = Base.summarysize(H) # size in bytes
    nr = length(H) *  sizeof(eltype(H))# represented entries
    return nr / ns
end

"""
    num_stored_elements(H::HMatrix)

The number of entries stored in the representation. Note that this is *not*
`length(H)`.
"""
function num_stored_elements(H::HMatrix)
    ns = 0 # stored entries
    for block in leaves(H)
        data = block.data
        ns += num_stored_elements(data)
    end
    return ns
end

num_stored_elements(M::Base.Matrix) = length(M)

function Base.show(io::IO, hmat::HMatrix)
    isclean(hmat) || return print(io, "Dirty HMatrix")
    print(io, "HMatrix of $(eltype(hmat)) with range $(rowrange(hmat)) × $(colrange(hmat))")
    _show(io, hmat, false)
    return io
end
Base.show(io::IO, ::MIME"text/plain", hmat::HMatrix) = show(io, hmat)

function _show(io, hmat, allow_empty = false)
    nodes_ = nodes(hmat)
    @printf io "\n\t number of nodes in tree: %i" length(nodes_)
    # filter empty leaves
    leaves_ = allow_empty ? filter(x->!isnothing(data(x)), leaves(hmat)) : leaves(hmat)
    sparse_leaves = filter(x->isadmissible(x), leaves_)
    dense_leaves  = filter(!isadmissible, leaves_)
    @printf(
        io,
        "\n\t number of leaves: %i (%i admissible + %i full)",
        length(leaves_),
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
    points_per_leaf = map(length, leaves_)
    @printf(io, "\n\t min number of elements per leaf: %i", minimum(points_per_leaf))
    @printf(io, "\n\t max number of elements per leaf: %i", maximum(points_per_leaf))
    depth_per_leaf = map(depth, leaves_)
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
    for block in nodes(hmat)
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
    assemble_hmatrix([T,], K, rowtree, coltree;
        adm=StrongAdmissibilityStd(),
        comp=PartialACA(),
        threads=true,
        distributed=false,
        global_index=true)

Main routine for assembling a hierarchical matrix. The argument `K` represents
the matrix to be approximated, `rowtree` and `coltree` are tree structure
partitioning the row and column indices, respectively, `adm` can be called on a
node of `rowtree` and a node of `coltree` to determine if the block is
compressible, and `comp` is a function/functor which can compress admissible
blocks.

It is assumed that `K` supports `getindex(K,i,j)`, and that `comp` can be called
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
        hmat_ = HMatrix{T}(rowtree, coltree, adm)
        # wrap hmat in a Hermitian if needed
        hmat = if K isa Hermitian
            Hermitian(hmat_)
        else
            hmat_
        end
        # if needed permute kernel entries into indexing induced by trees
        global_index && (K = PermutedMatrix(K, loc2glob(rowtree), loc2glob(coltree)))
        # now assemble the data in the blocks
        if threads
            _assemble_threads!(hmat, K, comp)
        else
            _assemble_cpu!(hmat, K, comp, ACABuffer(T))
        end
    end
end

function assemble_hmatrix(K::AbstractMatrix, args...; kwargs...)
    return assemble_hmatrix(eltype(K), K, args...; kwargs...)
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
        _process_leaf!(hmat, K, comp, bufs)
    else
        # recurse on children
        for child in children(hmat)
            _assemble_cpu!(child, K, comp, bufs)
        end
    end
    return hmat
end

function _process_leaf!(leaf, K, comp, bufs)
    # for hermitian, the upper trianglar stores the data
    (leaf isa Adjoint || leaf isa Transpose) && (return leaf)
    H = if leaf isa Hermitian
        parent(leaf) # the underlying HMatrix
    elseif leaf isa HMatrix
        leaf
    else
        throw(ArgumentError("Invalid leaf type: $(typeof(leaf))"))
    end
    #
    if isadmissible(H)
        _assemble_sparse_block!(H, K, comp, bufs)
    else
        _assemble_dense_block!(H, K)
    end
end

"""
    _assemble_threads!(hmat::HMatrix,K,comp)

Like [`_assemble_cpu!`](@ref), but uses threads to assemble the leaves. Note
that the threads are spanwned using `Threads.@spawn`, which means they are
spawned on the same worker as the caller.
"""
function _assemble_threads!(hmat, K, comp)
    c = leaves(hmat)
    acc = Threads.Atomic{Int}(1)
    # spawn np workers to assemble the leaves in parallel
    np = Threads.nthreads()
    @sync for _ in 1:np
        Threads.@spawn begin
            buf = ACABuffer(eltype(hmat))
            while true
                i = Threads.atomic_add!(acc, 1)
                i > length(c) && break
                _process_leaf!(c[i], K, comp, buf)
            end
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

# operation on adjoint
hasdata(adjH::Adjoint{<:Any,<:HMatrix}) = hasdata(adjH.parent)
data(adjH::Adjoint{<:Any,<:HMatrix}) = adjoint(data(adjH.parent))
children(adjH::Adjoint{<:Any,<:HMatrix}) = adjoint(children(adjH.parent))
pivot(adjH::Adjoint{<:Any,<:HMatrix}) = reverse(pivot(adjH.parent))
offset(adjH::Adjoint{<:Any,<:HMatrix}) = pivot(adjH) .- 1
rowrange(adjH::Adjoint{<:Any,<:HMatrix}) = colrange(adjH.parent)
colrange(adjH::Adjoint{<:Any,<:HMatrix}) = rowrange(adjH.parent)
isleaf(adjH::Adjoint{<:Any,<:HMatrix}) = isleaf(adjH.parent)
rowperm(adjH::Adjoint{<:Any,<:HMatrix}) = colperm(adjH.parent)
colperm(adjH::Adjoint{<:Any,<:HMatrix}) = rowperm(adjH.parent)
Base.size(adjH::Adjoint{<:Any,<:HMatrix}) = reverse(size(adjH.parent))

function Base.show(io::IO, adjH::Adjoint{<:Any,<:HMatrix})
    hmat = parent(adjH)
    isclean(hmat) || return print(io, "Dirty HMatrix")
    print(
        io,
        "Adjoint HMatrix of $(eltype(hmat)) with range $(rowrange(adjH)) × $(colrange(adjH))",
    )
    _show(io, hmat, false)
    return io
end
Base.show(io::IO, ::MIME"text/plain", adjH::Adjoint{<:Any,<:HMatrix}) = show(io, adjH)

# operations on hermitian
hasdata(hermH::Hermitian{<:Any,<:HMatrix})         = hasdata(hermH |> parent)
data(hermH::Hermitian{<:Any,<:HMatrix})            = data(hermH |> parent)
pivot(hermH::Hermitian{<:Any,<:HMatrix})           = hermH |> parent |> pivot
rowperm(hermH::Hermitian{<:Any,<:HMatrix})         = hermH |> parent |> rowperm
colperm(hermH::Hermitian{<:Any,<:HMatrix})         = hermH |> parent |> colperm
isleaf(hermH::Hermitian{<:Any,<:HMatrix})          = isleaf(parent(hermH))
rowrange(hermH::Hermitian{<:Any,<:HMatrix})        = hermH |> parent |> rowrange
colrange(hermH::Hermitian{<:Any,<:HMatrix})        = hermH |> parent |> colrange
Base.size(hermH::Hermitian{<:Any,<:HMatrix})       = hermH |> parent |> size
isadmissible(hermH::Hermitian) = isadmissible(hermH |> parent)

function children(hermH::Hermitian{<:Any,<:HMatrix})
    par_chd = hermH |> parent |> children
    return [children(hermH, i, j) for i in 1:size(par_chd, 1), j in 1:size(par_chd, 2)]
end
function children(hermH::Hermitian{<:Any,<:HMatrix}, i, j)
    par_chd = hermH |> parent |> children
    if i == j
        return Hermitian(par_chd[i, j])
    elseif i < j
        return par_chd[i, j]
    else
        return adjoint(par_chd[j, i])
    end
end

function Base.show(io::IO, hermH::Hermitian{<:Any,<:HMatrix})
    hmat = parent(hermH)
    print(
        io,
        "Hermitian HMatrix of $(eltype(hmat)) with range $(rowrange(hmat)) × $(colrange(hmat))",
    )
    _show(io, hmat, true)
    return io
end
Base.show(io::IO, ::MIME"text/plain", hermH::Hermitian{<:Any,<:HMatrix}) = show(io, hermH)

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
    for node in nodes(H)
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
    for leaf in leaves(H)
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
    for block in leaves(hmat)
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

# adding a sparse matrix to an HMatrix is allowed, but in the current
# implementation is done to recompress blocks which may increase rank during the
# process. If the rank increases by a large amount, we just print a warning for now.
function LinearAlgebra.axpy!(
    a,
    X::AbstractSparseArray{<:Any,<:Any,2},
    Y::HMatrix;
    global_index = true,
)
    rp = loc2glob(rowtree(Y))
    cp = loc2glob(coltree(Y))
    global_index && (X = permute(X, rp, cp))
    size_start = num_stored_elements(Y)
    _axpy!(a, X, Y)
    size_end = num_stored_elements(Y)
    size_end / size_start > 1.1 && @warn "Rank increased by more than 10% during axpy!"
    return Y
end

function _axpy!(a, X::AbstractSparseArray, Y::HMatrix)
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
                    if isadmissible(Y)
                        R = data(Y)
                        m, n = size(R)
                        # create a rank one matrix and add it to R
                        a = zeros(T, m)
                        b = zeros(T, n)
                        a[i-irange.start+1] = vals[idx]
                        b[j-jrange.start+1] = 1
                        R.A = hcat(R.A, a)
                        R.B = hcat(R.B, b)
                    else
                        M = data(Y)::Matrix{T} #
                        M[i-irange.start+1, j-jrange.start+1] += a * vals[idx]
                    end
                else
                    break # go to next column
                end
            end
        end
    else # has children
        for child in children(Y)
            _axpy!(a, X, child)
        end
    end
    return Y
end

Base.:(+)(X::AbstractSparseArray, Y::HMatrix) = axpy!(true, X, deepcopy(Y))
Base.:(+)(X::HMatrix, Y::AbstractSparseArray) = Y + X
