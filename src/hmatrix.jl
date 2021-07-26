"""
    mutable struct HMatrix{R,T}

A hierarchial matrix with constructure from a `rowtree` of type `R` and
`coltree` of type `R` holding elements of type `T`.
"""
mutable struct HMatrix{R,T}
    rowtree::R
    coltree::R
    admissible::Bool
    data::Union{Matrix{T},RkMatrix{T}}
    children::Matrix{HMatrix{R,T}}
    parent::HMatrix{R,T}
    # incomplete constructor.
    function HMatrix{R,T}(rowtree,coltree,adm,data,children,parent) where {R,T}
        hmat = new{R,T}(rowtree,coltree,adm)
        if data !== nothing
            @assert (length(rowtree),length(coltree)) === size(data) "$(length(rowtree)),$(length(coltree)) != $(size(data))"
            hmat.data = data
        end
        hmat.children = isnothing(children) ? Matrix{HMatrix{R,T}}(undef,0,0) : children
        hmat.parent   = isnothing(parent) ? hmat : parent
        return hmat
    end
end

# setters and getters
isleaf(H::HMatrix)                   = isempty(H.children)
isroot(H::HMatrix)                   = H === H.parent
isadmissible(H::HMatrix)             = H.admissible
hasdata(H::HMatrix)                  = isdefined(H,:data)

rowrange(H::HMatrix)         = range(H.rowtree)
colrange(H::HMatrix)         = range(H.coltree)
rowperm(H::HMatrix)          =  H.rowtree.loc2glob
colperm(H::HMatrix)          =  H.coltree.loc2glob
pivot(H::HMatrix)            = (rowrange(H).start,colrange(H).start)

Base.eltype(::HMatrix{R,T}) where {R,T} = T

Base.size(H::HMatrix) = length(rowrange(H)), length(colrange(H))
Base.size(H::HMatrix,i::Int) = size(H)[i]
Base.length(H::HMatrix) = prod(size(H))

idx_global_to_local(I,J,H::HMatrix) = (I,J) .- pivot(H) .+ 1

"""
    compression_ratio(H::HMatrix)

The ratio of the uncompressed size of `H` to its compressed size.
"""
function compression_ratio(H::HMatrix)
    ns = 0 # stored entries
    nr = length(H) # represented entries
    for block in AbstractTrees.Leaves(H)
        data = block.data
        ns  += num_stored_elements(data)
   end
   return nr/ns
end

num_stored_elements(M::Base.Matrix) = length(M)

# Interface to AbstractTrees.
# NOTE: for performance critical parts of the code, it is often better to
# recurse than to use this interface due to allocations incurred by the
# iterators. Maybe this can be fixed...
AbstractTrees.children(H::HMatrix) = isleaf(H) ? () : H.children
AbstractTrees.nodetype(H::HMatrix) = typeof(H)

function Base.show(io::IO,hmat::HMatrix)
    adm_str = hmat.admissible ? "admissible " : "non-admissible "
    print(io,adm_str*"hmatrix with range ($(rowrange(hmat))) × ($(colrange(hmat)))")
end

"""
    HMatrix([resource=CPU1()],K,blocktree,comp)

Main constructor for hierarchical matrix, where `K` represents the matrix to be
approximated, `blocktree` encondes the tree structure, and `comp` is a
function/functor which can compress admissible blocks.

It is assumed that `K` supports `getindex(K,i,j)`, that `blocktree` has methods
`getchildren(blocktree)`, `isadmissible(blocktree)`,
`rowrange(blocktree)->UnitRange`, and `colrange(blocktree)->UnitRange` , and
that comp can be called as `comp(K,irange::UnitRange,jrange::UnitRange)` to
produce a compressed version of `K[irange,jrange]`.

An optional first argument can be passed to control how the assembling is done.
The current options are:
- `::CPU1`
- `::CPUThreads`
"""
function HMatrix(resource::Union{AbstractResource},
                K,rowtree,coltree,adm,comp;
                permute_kernel=true)
    T  = eltype(K)
    # create first the structure
    @timeit_debug "initilizing block structure" begin
        hmat = HMatrix{T}(rowtree,coltree,adm)
    end

    # if needed permute kernel entries
    permute_kernel && (K = PermutedMatrix(K,rowtree.loc2glob,coltree.loc2glob))

    # now assemble the data in the blocks
    @timeit_debug "assembling hmatrix" begin
        assemble!(resource,hmat,K,comp) # recursive function
    end
end
#default
HMatrix(args...;kwargs...) = HMatrix(CPU1(),args...;kwargs...)

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
    # when a block has all of its children being non-admissible, aggreate them
    # into a bigger non-admissible block instead
    coarsen_non_admissible_blocks(root)
    return root
end

"""
    _build_block_structure!(adm_fun,current_node)

Recursive constructor for [`HMatrix`](@ref) block structure. Should not be called directly.
"""
function _build_block_structure!(adm, current_node::HMatrix{R,T}) where {R,T}
    X = current_node.rowtree
    Y = current_node.coltree
    if adm(X,Y)
        current_node.admissible = true
    else
        current_node.admissible = false
        if !(isleaf(X) || isleaf(Y))
            row_children = X.children
            col_children = Y.children
            children     = [HMatrix{R,T}(r,c,false,nothing,nothing,current_node) for r in row_children, c in col_children]
            current_node.children = children
            for child in children
                _build_block_structure!(adm,child)
            end
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
    assemble!(resource,hmat::HMatrix,K,comp)

Assemble data on the leaves of `hmat`. The admissible leaves are compressed
using the compressor `comp`. This function assumes the structure of `hmat` has
already been intialized, and therefore should rarely be called directly.
"""
function assemble!(resource::CPU1,hmat,K,comp)
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
            assemble!(resource,child,K,comp)
        end
    end
    return hmat
end

function assemble!(::CPUThreads,hmat,K,comp)
    # NOTE: ideally something like omp for schedule(guided) should be used here
    # to avoid spawning too many (small) tasks. In the absece of such scheduling
    # strategy in julia at the moment, we resort to manually limiting the size
    # of the tasks by directly calling the serial method for blocks which are
    # smaller than a given length (1000^2 here).
    blocks  = getnodes(x -> isleaf(x) || length(x)<1000*1000,hmat)
    sort!(blocks;lt=(x,y)->length(x)<length(y),rev=true)
    n = length(blocks)
    @sync begin
        for i in 1:n
            Threads.@spawn assemble!(CPU1(),blocks[i],K,comp)
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

"""
    struct StrongAdmissibilityStd

Two `ClusterTree`s are admissible under this condition if the minimum their
`diameter` is smaller than `eta` times the `distance` between the
them, where `eta::Float64` is a parameter.
"""
Base.@kwdef struct StrongAdmissibilityStd
    eta::Float64=3.0
end

function (adm::StrongAdmissibilityStd)(left_node::ClusterTree, right_node::ClusterTree)
    diam_min = minimum(diameter,(left_node,right_node))
    dist     = distance(left_node,right_node)
    return diam_min < adm.eta*dist
end

"""
    struct WeakAdmissibilityStd

Two `ClusterTree`s are admissible under this condition if the `distance`
between them is positive.
"""
struct WeakAdmissibilityStd
end

(adm::WeakAdmissibilityStd)(left_node::ClusterTree, right_node::ClusterTree) = distance(left_node,right_node) > 0

function Base.summary(io::IO,hmat::HMatrix)
    print("HMatrix{$(eltype(hmat))} spanning $(rowrange(hmat)) × $(colrange(hmat))")
    nodes = collect(AbstractTrees.PreOrderDFS(hmat))
    @printf "\n\t number of nodes in tree: %i" length(nodes)
    leaves = collect(AbstractTrees.Leaves(hmat))
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

@recipe function f(hmat::HMatrix)
    legend --> false
    grid   --> false
    aspect_ratio --> :equal
    yflip  := true
    seriestype := :shape
    linecolor  --> :black
    # all leaves
    for block in AbstractTrees.Leaves(hmat)
        @series begin
            if block.admissible
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
