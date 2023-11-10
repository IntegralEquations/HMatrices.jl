"""
    mutable struct ClusterTree{N,T}

Tree structure used to cluster poitns of type `SVector{N,T}` into [`HyperRectangle`](@ref)s.

# Fields:
- `_elements::Vector{SVector{N,T}}` : vector containing the sorted elements.
- `container::HyperRectangle{N,T}` : container for the elements in the current node.
- `index_range::UnitRange{Int}` : indices of elements contained in the current node.
- `loc2glob::Vector{Int}` : permutation from the local indexing system to the
  original (global) indexing system used as input in the construction of the
  tree.
- `glob2loc::Vector{Int}` : inverse of `loc2glob` permutation.
- `children::Vector{ClusterTree{N,T}}`
- `parent::ClusterTree{N,T}`
"""
mutable struct ClusterTree{N,T}
    _elements::Vector{SVector{N,T}}
    container::HyperRectangle{N,T}
    index_range::UnitRange{Int}
    loc2glob::Vector{Int}
    glob2loc::Vector{Int}
    children::Vector{ClusterTree{N,T}}
    parent::ClusterTree{N,T}
    function ClusterTree(
        els::Vector{SVector{N,T}},
        container,
        loc_idxs,
        loc2glob,
        glob2loc,
        children,
        parent,
    ) where {N,T}
        clt = new{N,T}(els, container, loc_idxs, loc2glob, glob2loc)
        clt.children = isnothing(children) ? Vector{typeof(clt)}() : children
        clt.parent = isnothing(parent) ? clt : parent
        return clt
    end
end

# convenience functions and getters
"""
    root_elements(clt::ClusterTree)

The elements contained in the root of the tree to which `clt` belongs.
"""
root_elements(clt::ClusterTree) = clt._elements

"""
    index_range(clt::ClusterTree)

Indices of elements in `root_elements(clt)` which lie inside `clt`.
"""
index_range(clt::ClusterTree) = clt.index_range

children(clt::ClusterTree) = clt.children

"""
    parent(t::ClusterTree)

The node's parent. If `t` is a root, then `parent(t)==t`.
"""
Base.parent(clt::ClusterTree) = clt.parent

"""
    container(clt::ClusterTree)

Return the object enclosing all the elements of the `clt`.
"""
container(clt::ClusterTree) = clt.container

"""
    elements(clt::ClusterTree)

Iterable list of the elements inside `clt`.
"""
elements(clt::ClusterTree) = view(root_elements(clt), index_range(clt))

"""
    loc2glob(clt::ClusterTree)

The permutation from the (local) indexing system of the elements of the `clt` to
the (global) indexes used upon the construction of the tree.
"""
loc2glob(clt::ClusterTree) = clt.loc2glob

"""
    glob2loc(clt::ClusterTree)

The inverse of [`loc2glob`](@ref).
"""
glob2loc(clt::ClusterTree) = clt.glob2loc

isleaf(clt::ClusterTree) = isempty(clt.children)
isroot(clt::ClusterTree) = clt.parent == clt

# for compatibility with AbstractTrees
AbstractTrees.children(t::ClusterTree) = isleaf(t) ? () : t.children
AbstractTrees.childrentype(::Type{T}) where {T<:ClusterTree} = Vector{T}

diameter(node::ClusterTree) = diameter(container(node))
radius(node::ClusterTree) = radius(container(node))

"""
    distance(X::ClusterTree, Y::ClusterTree)

Distance between the containers of `X` and `Y`.
"""
function distance(node1::ClusterTree, node2::ClusterTree)
    return distance(container(node1), container(node2))
end

Base.length(node::ClusterTree) = length(index_range(node))

"""
    ClusterTree(elements,splitter;[copy_elements=true, threads=false])

Construct a `ClusterTree` from the  given `elements` using the splitting
strategy encoded in `splitter`. If `copy_elements` is set to false, the
`elements` argument are directly stored in the `ClusterTree` and are permuted
during the tree construction.
"""
function ClusterTree(
    elements,
    splitter = CardinalitySplitter();
    copy_elements = true,
    threads = false,
)
    copy_elements && (elements = deepcopy(elements))
    bbox = bounding_box(elements)
    n = length(elements)
    irange = 1:n
    loc2glob = collect(irange)
    glob2loc = collect(irange) # used as buffer during the construction
    children = nothing
    parent = nothing
    #build the root, then recurse
    root = ClusterTree(elements, bbox, irange, loc2glob, glob2loc, children, parent)
    _build_cluster_tree!(root, splitter, threads)
    # inverse the loc2glob permutation
    glob2loc .= invperm(loc2glob)
    # finally, permute the elements so as to use the local indexing
    copy!(elements, elements[loc2glob]) # faster than permute!
    return root
end

function _build_cluster_tree!(current_node, splitter, threads, depth = 0)
    if should_split(current_node, depth, splitter)
        split!(current_node, splitter)
        if threads
            Threads.@threads for child in children(current_node)
                _build_cluster_tree!(child, splitter, threads, depth + 1)
            end
        else
            for child in children(current_node)
                _build_cluster_tree!(child, splitter, threads, depth + 1)
            end
        end
    end
    return current_node
end

function Base.show(io::IO, tree::ClusterTree{N,T}) where {N,T}
    return print(io, "ClusterTree with $(length(tree.index_range)) points.")
end

function Base.summary(clt::ClusterTree)
    @printf "Cluster tree with %i elements" length(clt)
    nodes = collect(AbstractTrees.PreOrderDFS(clt))
    @printf "\n\t number of nodes: %i" length(nodes)
    leaves = collect(AbstractTrees.Leaves(clt))
    @printf "\n\t number of leaves: %i" length(leaves)
    points_per_leaf = map(length, leaves)
    @printf "\n\t min number of elements per leaf: %i" minimum(points_per_leaf)
    @printf "\n\t max number of elements per leaf: %i" maximum(points_per_leaf)
    depth_per_leaf = map(depth, leaves)
    @printf "\n\t min depth of leaves: %i" minimum(depth_per_leaf)
    @printf "\n\t max depth of leaves: %i" maximum(depth_per_leaf)
end
