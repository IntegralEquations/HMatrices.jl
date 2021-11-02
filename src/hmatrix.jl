abstract type  AbstractHMatrix{T} <: AbstractMatrix{T} end

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
    # inner constructor which handles `nothing` fields.
    function HMatrix{R,T}(rowtree,coltree,adm,data,children,parent) where {R,T}
        if data !== nothing
            @assert (length(rowtree),length(coltree)) === size(data) "$(length(rowtree)),$(length(coltree)) != $(size(data))"
        end
        hmat = new{R,T}(rowtree,coltree,adm,data)
        hmat.children = isnothing(children) ? Matrix{HMatrix{R,T}}(undef,0,0) : children
        hmat.parent   = isnothing(parent) ? hmat : parent
        return hmat
    end
end

# setters and getters
isadmissible(H::HMatrix)  = H.admissible
hasdata(H::HMatrix)       = !isnothing(H.data)
data(H::HMatrix)          = H.data
setdata!(H::HMatrix,d) = setfield!(H,:data,d)
rowtree(H::HMatrix) = H.rowtree
coltree(H::HMatrix) = H.coltree

function Base.getindex(H::HMatrix,i::Int,j::Int)
    @debug "using `getindex(H::AbstractHMatrix,i::Int,j::Int)`."
    shift = pivot(H) .-1
    _getindex(H,i+shift[1],j+shift[2])
end

function _getindex(H,i,j)
    (i ∈ rowrange(H)) && (j ∈ colrange(H)) || throw(BoundsError(H,(i,j)))
    acc = zero(eltype(H))
    shift = pivot(H) .- 1
    if hasdata(H)
        il = i - shift[1]
        jl = j - shift[2]
        acc  += data(H)[il,jl]
    end
    for child in children(H)
        if (i ∈ rowrange(child)) && (j ∈ colrange(child))
            acc += _getindex(child,i,j)
        end
    end
    return acc
end

Base.getindex(H::HMatrix,::Colon,j) = getcol(H,j)

# getcol for regular matrices
function getcol!(col,M::Matrix,j)
    @assert length(col) == size(M,1)
    copyto!(col,view(M,:,j))
end
function getcol!(col,adjM::Adjoint{<:Any,<:Matrix},j)
    @assert length(col) == size(adjM,1)
    copyto!(col,view(adjM,:,j))
end

getcol(M::Matrix,j) = M[:,j]
getcol(adjM::Adjoint{<:Any,<:Matrix},j) = adjM[:,j]


function getcol(H::HMatrix,j::Int)
    m,n = size(H)
    T   = eltype(H)
    col = zeros(T,m)
    getcol!(col,H,j)
end

function getcol!(col,H::HMatrix,j::Int)
    (j ∈ colrange(H)) || throw(BoundsError())
    piv = pivot(H)
    _getcol!(col,H,j,piv)
end

function _getcol!(col,H::HMatrix,j,piv)
    if hasdata(H)
        shift        = pivot(H) .- 1
        jl           = j - shift[2]
        irange       = rowrange(H) .- (piv[1] - 1)
        getcol!(view(col,irange),data(H),jl)
    end
    for child in children(H)
        if j ∈ colrange(child)
            _getcol!(col,child,j,piv)
        end
    end
    return col
end

Base.getindex(adjH::Adjoint{<:Any,<:HMatrix},::Colon,j) = getcol(adjH,j)

function getcol(adjH::Adjoint{<:Any,<:HMatrix},j::Int)
    # (j ∈ colrange(adjH)) || throw(BoundsError())
    m,n = size(adjH)
    T   = eltype(adjH)
    col = zeros(T,m)
    getcol!(col,adjH,j)
end

function getcol!(col,adjH::Adjoint{<:Any,<:HMatrix},j::Int)
    piv = pivot(adjH)
    _getcol!(col,adjH,j,piv)
end

function _getcol!(col,adjH::Adjoint{<:Any,<:HMatrix},j,piv)
    if hasdata(adjH)
        shift        = pivot(adjH) .- 1
        jl           = j - shift[2]
        irange       = rowrange(adjH) .- (piv[1] - 1)
        getcol!(view(col,irange),data(adjH),jl)
    end
    for child in children(adjH)
        if j ∈ colrange(child)
            _getcol!(col,child,j,piv)
        end
    end
    return col
end

# Trees interface
Trees.children(H::HMatrix) = H.children
Trees.children(H::HMatrix,idxs...) = H.children[idxs]
Trees.parent(H::HMatrix)   = H.parent
Trees.isleaf(H::HMatrix)   = isempty(children(H))
Trees.isroot(H::HMatrix)   = parent(H) === H

# interface to AbstractTrees. No children is determined by an empty tuple for
# AbstractTrees.
AbstractTrees.children(t::HMatrix) = isleaf(t) ? () : t.children
AbstractTrees.nodetype(t::HMatrix) = typeof(t)

rowrange(H::HMatrix)         = Trees.index_range(H.rowtree)
colrange(H::HMatrix)         = Trees.index_range(H.coltree)
rowperm(H::HMatrix)          =  H |> rowtree |> loc2glob
colperm(H::HMatrix)          =  H |> coltree |> loc2glob
pivot(H::HMatrix)            = (rowrange(H).start,colrange(H).start)
offset(H::HMatrix)           = pivot(H) .- 1

# Base.axes(H::HMatrix) = rowrange(H),colrange(H)
Base.size(H::HMatrix) = length(rowrange(H)), length(colrange(H))

function blocksize(H::HMatrix)
    size(children(H))
end

"""
    compression_ratio(H::HMatrix)

The ratio of the uncompressed size of `H` to its compressed size.
"""
function compression_ratio(H::HMatrix)
    ns = 0 # stored entries
    nr = length(H) # represented entries
    for block in Leaves(H)
        data = block.data
        ns  += num_stored_elements(data)
   end
   return nr/ns
end

num_stored_elements(M::Base.Matrix) = length(M)

function Base.show(io::IO,::MIME"text/plain",hmat::HMatrix)
    adm_str = hmat.admissible ? "admissible " : "non-admissible "
    print(io,adm_str*"hmatrix with range ($(rowrange(hmat))) × ($(colrange(hmat)))")
end

"""
    Matrix(H::HMatrix;global_index=false)

Convert `H` to a `Matrix`. If `global_index=true`, the entries are given in the
global indexing system (see [`HMatrix`](@ref) for more information); otherwise
the *local* indexing system induced by the row and columns trees are used
(default).
"""
Matrix(hmat::HMatrix;global_index=false) = Matrix{eltype(hmat)}(hmat;global_index)
function Matrix{T}(hmat::HMatrix;global_index) where {T}
    M = zeros(T,size(hmat)...)
    piv = pivot(hmat)
    for block in PreOrderDFS(hmat)
        hasdata(block) || continue
        irange = rowrange(block) .- piv[1] .+ 1
        jrange = colrange(block) .- piv[2] .+ 1
        M[irange,jrange] += Matrix(block.data)
    end
    if global_index
        P = PermutedMatrix(M,invperm(rowperm(hmat)),invperm(colperm(hmat)))
        return Matrix(P)
    else
        return M
    end
end

"""
    HMatrix(K,rowtree,coltree,adm,comp;threads=true,distributed=false,permute_kernel=true)

Main constructor for hierarchical matrix, where `K` represents the matrix to be
approximated, `rowtree` and `coltree` are tree structure partitioning the row
and column indices, respectively, `adm` can be called on a node of `rowtree` and
a node of `coltree` to determine if the block is compressible, and `comp` is a
function/functor which can compress admissible blocks.

It is assumed that `K` supports `getindex(K,i,j)`, and that comp can be called
as `comp(K,irange::UnitRange,jrange::UnitRange)` to produce a compressed version
of `K[irange,jrange]`.
"""
function HMatrix(K,rowtree,coltree,
                adm=StrongAdmissibilityStd(3),
                comp=PartialACA(;rtol=1e-6);
                permute_kernel=true,
                threads=true,
                distributed=false)
    T  = eltype(K)
    # create first the structure. No parellelism used as this should be light.
    @timeit_debug "initilizing block structure" begin
        hmat = HMatrix{T}(rowtree,coltree,adm)
    end

    # if needed permute kernel entries into indexing induced by trees
    permute_kernel && (K = PermutedMatrix(K,loc2glob(rowtree),loc2glob(coltree)))

    # now assemble the data in the blocks
    @timeit_debug "assembling hmatrix" begin
        if distributed
            error("distributed assembly not yet supported")
        else
            if threads
                assemble_threads!(hmat,K,comp)
            else
                assemble_cpu!(hmat,K,comp)
            end
        end
    end
end

"""
    HMatrix{T}(rowtree,coltree,adm)

Construct an empty `HMatrix` with `rowtree` and `coltree` using the
admissibility condition `adm`.
"""
function HMatrix{T}(rowtree::R, coltree::R, adm) where {R,T}
    #build root
    root  = HMatrix{R,T}(rowtree,coltree,false,nothing,nothing,nothing)
    # recurse
    _build_block_structure!(adm,root)
    # TODO: when a block has all of its children being non-admissible, aggregate them
    # into a bigger non-admissible block instead? The issue is that this
    # destroys some of the reasoning behind which of the 27 mul! methods are
    # actually reachable in the "regular" case.
    # coarsen_non_admissible_blocks(root)
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
    elseif adm(X,Y)
        current_node.admissible = true
    else
        current_node.admissible = false
        row_children = X.children
        col_children = Y.children
        children     = [HMatrix{R,T}(r,c,false,nothing,nothing,current_node) for r in row_children, c in col_children]
        current_node.children = children
        for child in children
            _build_block_structure!(adm,child)
        end
    end
    return current_node
end

"""
    coarsen_non_admissible_blocks(H::HMatrix)

Can be called after initializing the structure of an `HMatrix` to eliminate all
non-admissible leaves for which all siblings are also non-admissible leaves.
This has the effect of aggregating small dense blocks into larger ones.
"""
function coarsen_non_admissible_blocks(block)
    isleaf(block) && (return block)
    isvalid = all(block.children) do child
        isleaf(child) && !isadmissible(child)
    end
    if isvalid
        block.children   = Matrix{typeof(block)}(undef,0,0)
        block.admissible = false
        isroot(block) || coarsen_non_admissible_blocks(block.parent)
    else
        for  child in block.children
            coarsen_non_admissible_blocks(child)
        end
    end
    return block
end

"""
    assemble_cpu!(hmat::HMatrix,K,comp)

Assemble data on the leaves of `hmat`. The admissible leaves are compressed
using the compressor `comp`. This function assumes the structure of `hmat` has
already been intialized, and therefore should not be called directly. See
[`HMatrix`](@ref) information on constructors.
"""
function assemble_cpu!(hmat,K,comp)
    # base case
    if isleaf(hmat)
        if isadmissible(hmat)
            _assemble_sparse_block!(hmat,K,comp)
        else
            _assemble_dense_block!(hmat,K)
        end
    else
    # recurse on children
        for child in hmat.children
            assemble_cpu!(child,K,comp)
        end
    end
    return hmat
end

"""
    assemble_threads!(hmat::HMatrix,K,comp)

Like [`assemble_cpu!`](@ref), but uses threads to assemble (independent) blocks.
"""
function assemble_threads!(hmat,K,comp)
    # FIXME: ideally something like `omp for schedule(guided)` should be used here
    # to avoid spawning too many (small) tasks. In the absece of such scheduling
    # strategy in julia at the moment, we resort to manually limiting the size
    # of the tasks by directly calling the serial method for blocks which are
    # smaller than a given length (1000^2 here).
    filter = (x) -> !(isleaf(x) || length(x)<1000*1000)
    blocks = PreOrderDFS(hmat,filter) |> collect
    sort!(blocks;lt=(x,y)->length(x)<length(y),rev=true)
    n = length(blocks)
    @sync begin
        for i in 1:n
            Threads.@spawn assemble_cpu!(blocks[i],K,comp)
        end
    end
    return hmat
end

function _assemble_sparse_block!(hmat,K,comp)
    hmat.data = comp(K,hmat.rowtree,hmat.coltree)
end

function _assemble_dense_block!(hmat,K)
    irange = rowrange(hmat)
    jrange = colrange(hmat)
    hmat.data = K[irange,jrange]
    return hmat
end

# LinearAlgebra.adjoint(H::HMatrix) = Adjoint(H)
hasdata(adjH::Adjoint{<:Any,<:HMatrix}) = hasdata(adjH.parent)
data(adjH::Adjoint{<:Any,<:HMatrix}) = adjoint(data(adjH.parent))
Trees.children(adjH::Adjoint{<:Any,<:HMatrix}) = adjoint(children(adjH.parent))
pivot(adjH::Adjoint{<:Any,<:HMatrix}) = reverse(pivot(adjH.parent))
offset(adjH::Adjoint{<:Any,<:HMatrix}) = pivot(adjH) .- 1
rowrange(adjH::Adjoint{<:Any,<:HMatrix}) = colrange(adjH.parent)
colrange(adjH::Adjoint{<:Any,<:HMatrix}) = rowrange(adjH.parent)
Trees.isleaf(adjH::Adjoint{<:Any,<:HMatrix}) = isleaf(adjH.parent)

Base.size(adjH::Adjoint{<:Any,<:HMatrix}) = reverse(size(adjH.parent))

function Base.show(io::IO,::MIME"text/plain",hmat::Adjoint{Float64,<:HMatrix})
    print(io,"adjoint hmatrix with range ($(rowrange(hmat))) × ($(colrange(hmat)))")
end

"""
    struct StrongAdmissibilityStd

Two blocks are admissible under this condition if the minimum their
`diameter` is smaller than `eta` times the `distance` between the
them, where `eta::Float64` is a parameter.
"""
Base.@kwdef struct StrongAdmissibilityStd
    eta::Float64=3.0
end

function (adm::StrongAdmissibilityStd)(left_node, right_node)
    diam_min = minimum(diameter,(left_node,right_node))
    dist     = distance(left_node,right_node)
    return diam_min < adm.eta*dist
end

"""
    struct WeakAdmissibilityStd

Two blocks are admissible under this condition if the `distance`
between them is positive.
"""
struct WeakAdmissibilityStd
end

(adm::WeakAdmissibilityStd)(left_node, right_node) = distance(left_node,right_node) > 0

function Base.summary(io::IO,hmat::HMatrix)
    print("HMatrix{$(eltype(hmat))} spanning $(rowrange(hmat)) × $(colrange(hmat))")
    nodes = collect(PreOrderDFS(hmat))
    @printf "\n\t number of nodes in tree: %i" length(nodes)
    leaves = collect(Leaves(hmat))
    sparse_leaves = filter(isadmissible,leaves)
    dense_leaves  = filter(!isadmissible,leaves)
    @printf("\n\t number of leaves: %i (%i admissible + %i full)",length(leaves),length(sparse_leaves),
    length(dense_leaves))
    rmin,rmax = isempty(sparse_leaves) ? (-1,-1) : extrema(x->rank(x.data),sparse_leaves)
    @printf("\n\t minimum rank of sparse blocks : %i",rmin)
    @printf("\n\t maximum rank of sparse blocks : %i",rmax)
    dense_min,dense_max = isempty(dense_leaves) ? (-1,-1) : extrema(x->length(x.data),dense_leaves)
    @printf("\n\t minimum length of dense blocks : %i",dense_min)
    @printf("\n\t maximum length of dense blocks : %i",dense_max)
    points_per_leaf = map(length,leaves)
    @printf "\n\t min number of elements per leaf: %i" minimum(points_per_leaf)
    @printf "\n\t max number of elements per leaf: %i" maximum(points_per_leaf)
    depth_per_leaf = map(depth,leaves)
    @printf "\n\t depth of tree: %i" maximum(depth_per_leaf)
    @printf "\n\t compression ratio: %f\n" compression_ratio(hmat)
end

"""
    isclean(H::HMatrix)

Return `true` if `H` all leaves of `H` have data, and if the leaves are the only
nodes containing data. This is the normal state of an ℋ-matrix, but during
intermediate stages of a computation data may be associated with non-leaf nodes
for convenience.
"""
function isclean(H::HMatrix)
    for node in PreOrderDFS(H)
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

function depth(tree::HMatrix,acc=0)
    if isroot(tree)
        return acc
    else
        depth(parent(tree),acc+1)
    end
end

function Base.zero(H::HMatrix)
    H0 = deepcopy(H)
    rmul!(H0,0)
    return H0
end

function compress!(H::HMatrix,comp)
    @assert isclean(H)
    for leaf in Leaves(H)
        d = data(leaf)
        if d isa RkMatrix
            compress!(d,comp)
        end
    end
    return H
end


############################################################################################
# Recipes
############################################################################################

@recipe function f(hmat::HMatrix)
    legend --> false
    grid   --> false
    aspect_ratio --> :equal
    yflip  := true
    seriestype := :shape
    linecolor  --> :black
    # all leaves
    for block in Leaves(hmat)
        @series begin
            if isadmissible(block)
                fillcolor    --> :blue
                seriesalpha  --> 1/compression_ratio(block.data)
            else
                fillcolor    --> :red
                seriesalpha  --> 0.3
            end
            pt1 = pivot(block)
            pt2 = pt1 .+ size(block) .- 1
            y1, y2 = pt1[1],pt2[1]
            x1, x2 = pt1[2],pt2[2]
            # annotations := ((x1+x2)/2,(y1+y2)/2, rank(block.data))
            [x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1]
        end
    end
end
