"""
    mutable struct HMatrix{R,T} <: AbstractMatrix{T}

A hierarchical matrix constructed from a `rowtree` and `coltree` of type `R` and
holding elements of type `T`.
"""
mutable struct HMatrix{R, T} <: AbstractMatrix{T}
    rowtree::R
    coltree::R
    admissible::Bool
    data::Union{Matrix{T}, RkMatrix{T}, Nothing}
    children::Matrix{HMatrix{R, T}}
    # NOTE: to avoid confusion with Base.parent, which returns the underlying
    # parent object of a view, we use `parentnode` to refer to the tree parent.
    parentnode::HMatrix{R, T}
    # inner constructor which handles `nothing` fields.
    function HMatrix{R, T}(rowtree, coltree, adm, data, children, parentnode) where {R, T}
        if data !== nothing
            @assert (length(rowtree), length(coltree)) === size(data) "$(length(rowtree)),$(length(coltree)) != $(size(data))"
        end
        hmat = new{R, T}(rowtree, coltree, adm, data)
        hmat.children = isnothing(children) ? Matrix{HMatrix{R, T}}(undef, 0, 0) : children
        hmat.parentnode = isnothing(parentnode) ? hmat : parentnode
        return hmat
    end
end

# setters and getters defined for HMatrix.
isadmissible(H::HMatrix) = H.admissible
data(H::HMatrix) = H.data
rowtree(H::HMatrix) = H.rowtree
coltree(H::HMatrix) = H.coltree
children(H::HMatrix) = H.children
parentnode(H::HMatrix) = H.parentnode
setdata!(H::HMatrix, d) = setfield!(H, :data, d)

# light wrapper for adjoint
const AdjointHMatrix{R, T} = Adjoint{T, HMatrix{R, T}}

isadmissible(H::AdjointHMatrix) = H |> parent |> isadmissible
data(H::AdjointHMatrix) = H |> parent |> data |> adjoint
rowtree(H::AdjointHMatrix) = H |> parent |> coltree
coltree(H::AdjointHMatrix) = H |> parent |> rowtree
children(H::AdjointHMatrix) = H |> parent |> children |> adjoint
parentnode(H::AdjointHMatrix) = H |> parent |> parentnode |> adjoint

# light wrapper for Hermitian
const HermitianHMatrix{R, T} = Hermitian{T, HMatrix{R, T}}

isadmissible(H::HermitianHMatrix) = H |> parent |> isadmissible
data(H::HermitianHMatrix) = H |> parent |> data |> Hermitian
rowtree(H::HermitianHMatrix) = H |> parent |> rowtree
coltree(H::HermitianHMatrix) = H |> parent |> coltree
children(H::HermitianHMatrix) = H |> parent |> children |> Hermitian
parentnode(H::HermitianHMatrix) = H |> parent |> parentnode |> Hermitian

const HTypes = Union{HMatrix, AdjointHMatrix, HermitianHMatrix}

# forward definition of triangular types
const HLowerTriangular =
    Union{UnitLowerTriangular{<:Any, <:HTypes}, LowerTriangular{<:Any, <:HTypes}}
const HUpperTriangular =
    Union{UnitUpperTriangular{<:Any, <:HTypes}, UpperTriangular{<:Any, <:HTypes}}
const HTriangular = Union{HLowerTriangular, HUpperTriangular}

# NOTE: parent here refers to the underlying data of the view, NOT the
# parentnode. Ducktype and hope for the best.
hasdata(H) = !isnothing(data(parent(H)))
isroot(H) = parentnode(H) === H

# operations in the user ordering need the permutations of the whole matrix
function _checkroot(H)
    isroot(parent(H)) ||
        throw(ArgumentError("this operation is only defined on a whole `HMatrix`, not on one of its blocks"))
    return H
end
rowrange(H) = index_range(rowtree(H))
colrange(H) = index_range(coltree(H))
"""
    rowperm(H)

The permutation of the rows of the whole matrix into the ordering in which it is stored:
for the root `H` of an `HMatrix`, the stored row `i` is the row `rowperm(H)[i]` of the
matrix approximated by `H`. For a block of `H`, it is the permutation of the whole matrix
(the rows of the block are `rowperm(H)[rowrange(H)]`).

The returned vector belongs to the cluster tree and must not be modified.
"""
rowperm(H) = loc2glob(rowtree(H))

"""
    colperm(H)

Like [`rowperm`](@ref), for the columns of `H`.
"""
colperm(H) = loc2glob(coltree(H))
pivot(H) = (rowrange(H).start, colrange(H).start)
offset(H) = pivot(H) .- 1
Base.size(H::HTypes) = length(rowrange(H)), length(colrange(H))
blocksize(H) = H |> parent |> children |> size
isleaf(H) = H |> children |> isempty

function Base.getindex(::HTypes, i::Integer, j::Integer)
    msg = """method `getindex(::HMatrix,args...)` has been disabled to
    avoid performance pitfalls. Unless you made an explicit call to `getindex`,
    this error usually means that a linear algebra routine involving an
    `HMatrix` has fallen back to a generic implementation."""
    return error(msg)
end

# overload the getcol method
function getcol!(col, H::HTypes, j::Int)
    (j ∈ colrange(H)) || throw(BoundsError())
    piv = pivot(H)
    return _getcol!(col, H, j, piv)
end

function _getcol!(col, H::HTypes, j, piv)
    if hasdata(H)
        jl = j - offset(H)[2]
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

"""
    compression_ratio(H::HTypes)

The ratio of the uncompressed size of `H` to its compressed size. A
`compression_ratio` of `10` means it would have taken 10 times more memory to
store `H` as a dense matrix.
"""
function compression_ratio(H::HTypes)
    ns = Base.summarysize(H) # size in bytes
    nr = length(H) * sizeof(eltype(H)) # represented entries
    return nr / ns
end

function Base.show(io::IO, H::HTypes)
    hmat = parent(H)
    # the data of a `Hermitian` is stored in one triangle only, so its leaves may be empty
    herm = H isa Hermitian
    herm || isclean(hmat) || return print(io, "Dirty HMatrix")
    label = herm ? "Hermitian HMatrix" : H isa Adjoint ? "Adjoint HMatrix" : "HMatrix"
    print(io, "$label of $(eltype(hmat)) with range $(rowrange(H)) × $(colrange(H))")
    _show(io, hmat, herm)
    return io
end
Base.show(io::IO, ::MIME"text/plain", H::HTypes) = show(io, H)

function _show(io, hmat, allow_empty = false)
    nodes_ = nodes(hmat)
    @printf io "\n\t number of nodes in tree: %i" length(nodes_)
    # filter empty leaves
    leaves_ = allow_empty ? filter(x -> !isnothing(data(x)), leaves(hmat)) : leaves(hmat)
    sparse_leaves = filter(x -> isadmissible(x), leaves_)
    dense_leaves = filter(!isadmissible, leaves_)
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
    Matrix(H::HMatrix)

Convert `H` to a dense `Matrix`, in the ordering of the points used to assemble it.
"""
Base.Matrix(hmat::HTypes) = Matrix{eltype(hmat)}(hmat)
function Base.Matrix{T}(hmat::HTypes) where {T}
    _checkroot(hmat)
    M = _tree_matrix(hmat, T)
    return M[invperm(rowperm(hmat)), invperm(colperm(hmat))]
end

"""
    _tree_matrix(H, T = eltype(H))

The dense `Matrix{T}` of `H` in the ordering of its cluster trees, i.e. the ordering in
which `H` is stored.
"""
function _tree_matrix(hmat, T = eltype(hmat))
    M = zeros(T, size(hmat)...)
    piv = pivot(hmat)
    for leaf in leaves(hmat)
        irange = rowrange(leaf) .- piv[1] .+ 1
        jrange = colrange(leaf) .- piv[2] .+ 1
        M[irange, jrange] = Matrix(data(leaf))
    end
    return M
end

"""
    assemble_hmatrix(K, X, Y = X; atol = 0, rtol, rank, eta = 3, nmax = 50, threads = true)

Approximate the matrix `K` by an [`HMatrix`](@ref), where `X` and `Y` are the points
(`SVector`s, or elements with a [`center`](@ref)) associated with the rows and the
columns of `K`. The points are only used to cluster the indices, and are not modified.

- `K` can be any `AbstractMatrix` (e.g. a [`KernelMatrix`](@ref)), with scalar or
  `SMatrix` entries; [`getblock!`](@ref) can be overloaded to compute its blocks faster.
  If `K` is a `Hermitian`, only its upper triangle is assembled and a `Hermitian`
  `HMatrix` is returned (this requires `Y === X`).
- `atol`, `rtol` and `rank` control the compression of the low-rank blocks (see
  [`PartialACA`](@ref)); by default `rtol = sqrt(eps)` of the real type of the entries,
  or only the accuracy attainable with that type if `atol` or `rank` is given.
- Two clusters are compressed if `min(diameters) < eta * distance`; `eta = Inf`
  compresses every pair of disjoint clusters (weak admissibility).
- `nmax` is the maximum number of points in a leaf of the cluster trees.

Every operation on the result (products, `Matrix`, factorizations) uses the ordering of
`X` and `Y`.
"""
function assemble_hmatrix(
        K::AbstractMatrix,
        X::AbstractVector,
        Y::AbstractVector = X;
        atol = 0,
        rtol = nothing,
        rank = typemax(Int),
        eta = 3,
        nmax = 50,
        threads = true,
    )
    size(K) == (length(X), length(Y)) ||
        throw(DimensionMismatch("the matrix is of size $(size(K)), but there are $(length(X)) row points and $(length(Y)) column points"))
    K isa Hermitian && Y !== X &&
        throw(ArgumentError("a `Hermitian` matrix needs the same points for its rows and columns"))
    splitter = GeometricSplitter(; nmax)
    # `map` copies the points, which the trees then permute
    Xclt = ClusterTree(map(x -> float(center(x)), X), splitter; copy_elements = false)
    Yclt = Y === X ? Xclt : ClusterTree(map(y -> float(center(y)), Y), splitter; copy_elements = false)
    adm = StrongAdmissibilityStd(; eta)
    comp = PartialACA(; atol, rtol, rank)
    return assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads)
end

"""
    assemble_hmatrix(K, rowtree::ClusterTree, coltree::ClusterTree;
        adm = StrongAdmissibilityStd(), comp = PartialACA(), threads = true)

Lower-level version of [`assemble_hmatrix`](@ref) with given cluster trees, admissibility
condition `adm` (called on a node of `rowtree` and a node of `coltree` to decide if the
block is compressed) and compressor `comp` (see [`AbstractCompressor`](@ref)).
"""
function assemble_hmatrix(
        K::AbstractMatrix,
        rowtree::ClusterTree,
        coltree::ClusterTree;
        adm = StrongAdmissibilityStd(),
        comp = PartialACA(),
        threads = true,
    )
    size(K) == (length(rowtree), length(coltree)) ||
        throw(DimensionMismatch("the matrix is of size $(size(K)), but the trees have $(length(rowtree)) and $(length(coltree)) elements"))
    T = eltype(K)
    isconcretetype(T) || throw(
        ArgumentError(
            "the element type of the matrix, $T, is not concrete; for a `KernelMatrix`, make sure that the kernel function is type stable (e.g. that it does not capture non-constant global variables)",
        ),
    )
    # create first the structure. No parallelism used as this should be light.
    hmat_ = HMatrix{T}(rowtree, coltree, adm)
    # wrap hmat in a Hermitian if needed
    hmat = if K isa Hermitian
        Hermitian(hmat_)
    else
        hmat_
    end
    # the blocks are assembled in the ordering of the trees
    K = PermutedMatrix(K, loc2glob(rowtree), loc2glob(coltree))
    # now assemble the data in the blocks
    if threads
        _assemble_threads!(hmat, K, comp)
    else
        _assemble_cpu!(hmat, K, comp, allocate_buffer(comp, T))
    end
    return hmat
end

"""
    HMatrix{T}(rowtree,coltree,adm)

Construct an empty `HMatrix` with `rowtree` and `coltree` using the
admissibility condition `adm`. This function builds the skeleton for the
hierarchical matrix, but **does not compute `data`** field in the blocks. See
[`assemble_hmatrix`](@ref) for assembling a hierarchical matrix.
"""
function HMatrix{T}(rowtree::R, coltree::R, adm) where {R, T}
    #build root
    root = HMatrix{R, T}(rowtree, coltree, false, nothing, nothing, nothing)
    # recurse
    _build_block_structure!(adm, root)
    return root
end

"""
    _build_block_structure!(adm_fun,current_node)

Recursive constructor for [`HMatrix`](@ref) block structure. Should not be called directly.
"""
function _build_block_structure!(adm, current_node::HMatrix{R, T}) where {R, T}
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
            HMatrix{R, T}(r, c, false, nothing, nothing, current_node) for
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
already been initialized, and therefore should not be called directly. See
[`HMatrix`](@ref) for information on constructors.
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
    # for hermitian, the upper triangular stores the data
    (leaf isa Adjoint || leaf isa Transpose) && (return leaf)
    H = if leaf isa Hermitian
        parent(leaf) # the underlying HMatrix
    elseif leaf isa HMatrix
        leaf
    else
        throw(ArgumentError("Invalid leaf type: $(typeof(leaf))"))
    end
    #
    return if isadmissible(H)
        _assemble_sparse_block!(H, K, comp, bufs)
    else
        _assemble_dense_block!(H, K)
    end
end

"""
    _assemble_threads!(hmat::HMatrix,K,comp)

Like [`_assemble_cpu!`](@ref), but uses threads to assemble the leaves. Note
that the threads are spawned using `Threads.@spawn`, which means they are
spawned on the same worker as the caller.
"""
function _assemble_threads!(hmat, K, comp)
    c = leaves(hmat)
    acc = Threads.Atomic{Int}(1)
    # spawn np workers to assemble the leaves in parallel
    np = Threads.nthreads()
    @sync for _ in 1:np
        Threads.@spawn begin
            buf = allocate_buffer(comp, eltype(hmat))
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
    return hmat.data = _compress(comp, K, hmat.rowtree, hmat.coltree, bufs)
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
@recipe function f(hmat::HTypes)
    legend --> false
    grid --> false
    aspect_ratio --> :equal
    yflip := true
    seriestype := :shape
    linecolor --> :black
    ylims --> (1, length(rowtree(hmat)))
    xlims --> (1, length(coltree(hmat)))
    for block in leaves(hmat)
        mat = data(block)
        @series begin
            if mat isa Adjoint # data is stored elsewhere, make it transparent
                fillcolor --> :lightgray
                seriesalpha --> 0.1
            elseif isadmissible(block)
                fillcolor --> :blue
                r = rank(mat)
                m, n = size(mat)
                alpha = m * n / (r * (m + n))
                seriesalpha --> 1 / alpha
            else
                fillcolor --> :red
                seriesalpha --> 0.3
            end
            pt1 = pivot(block)
            pt2 = pt1 .+ size(block) .- 1
            y1, y2 = pt1[1], pt2[1]
            x1, x2 = pt1[2], pt2[2]
            [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1]
        end
    end
end

# add a uniform scaling to an HMatrix return an HMatrix
function LinearAlgebra.axpy!(a, X::UniformScaling, Y::HMatrix)
    LinearAlgebra.checksquare(Y)
    # the identity is diagonal in the stored ordering only if rows and columns are
    # permuted in the same way
    rowperm(Y) == colperm(Y) || return axpy!(a, sparse(X, size(Y)...), Y)
    return _axpy_diagonal!(a, X, Y)
end

# add `a*X` to the diagonal of the stored blocks of `Y`
function _axpy_diagonal!(a, X::UniformScaling, Y::HMatrix)
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
            _axpy_diagonal!(a, X, children(Y)[i, i])
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
        X::AbstractSparseArray{<:Any, <:Any, 2},
        Y::HMatrix,
    )
    X = permute(X, rowperm(Y), colperm(Y))
    size_start = Base.summarysize(Y)
    _axpy!(a, X, Y)
    size_end = Base.summarysize(Y)
    size_end / size_start > 1.1 &&
        @warn "Memory size increased by more than 10% during axpy!"
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
                        u = zeros(T, m)
                        v = zeros(T, n)
                        u[i - irange.start + 1] = a * vals[idx]
                        v[j - jrange.start + 1] = one(T)
                        R.A = hcat(R.A, u)
                        R.B = hcat(R.B, v)
                    else
                        M = data(Y)::Matrix{T} #
                        M[i - irange.start + 1, j - jrange.start + 1] += a * vals[idx]
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
