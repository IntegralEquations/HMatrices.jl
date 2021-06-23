"""
    BlockTree{T,S}

Represents a block tree constructed from two [`ClusterTree{T,S}`](@ref) objects.
Determines the block structure of a `HierarchicalMatrix` without the actual
numbers.

Each node of the `BlockTree` contains a reference to the `row_cluster` and the
`col_cluster` which define the `BlockTree`. It also explicitly stores its
`parent` and `children`, together with a field `admissible` which determines
whether the current block is amenable to a low-rank approximation (see
[`AbstractAdmissibilityCondition`](@ref)).
"""
mutable struct BlockTree{T,S}
    row_cluster::T
    col_cluster::S
    admissible::Bool
    children::Maybe{Matrix{BlockTree{T,S}}}
    parent::Maybe{BlockTree{T,S}}
end

# interface to AbstractTrees
AbstractTrees.children(clt::BlockTree) = getchildren(clt)

Base.size(tree::BlockTree)   = (length(tree.row_cluster), length(tree.col_cluster))
Base.size(tree::BlockTree,i) = size(tree)[i]
Base.length(tree::BlockTree) = prod(size(tree))

rowrange(block::BlockTree)   = range(rowcluster(block))
colrange(block::BlockTree)   = range(colcluster(block))

pivot(block::BlockTree)      = (rowrange(block).start,colrange(block).start)

getchildren(bclt::BlockTree)           = bclt.children
getparent(bclt::BlockTree)             = bclt.parent
setchildren!(bclt::BlockTree,children) = (bclt.children = children)
setparent!(bclt::BlockTree,parent)     = (bclt.parent   = parent)

isleaf(clt::BlockTree)       = getchildren(clt) === ()
isroot(clt::BlockTree)       = getparent(clt)   === ()
isadmissible(clt::BlockTree) = clt.admissible

rowcluster(bclt::BlockTree) = bclt.row_cluster
colcluster(bclt::BlockTree) = bclt.col_cluster

"""
    abstract type AbstractAdmissibilityCondition

An `AbstractAdmissibilityCondition` is used to determine whether a
[`BlockTree`](@ref) node admits a low-rank approximation. Objects of this type
are used as functors and require implementing the following method

`(adm::AbstractAdmissibilityCondition)(X::ClusterTree,Y::ClusterTree)::Bool`

See (`StrongAdmissibiltyStd`)[@ref] for an example of a concrete implementation.
"""
abstract type AbstractAdmissibilityCondition end

(adm::AbstractAdmissibilityCondition)(bclt::BlockTree) = adm(rowcluster(bclt),colcluster(bclt))

"""
    struct StrongAdmissibilityStd <: AbstractAdmissibilityCondition

A `BlockTree` is admissible under this condition if the minimum of the
`diameter` of its `ClusterTree`s is smaller than `eta` times the `distance`
between the `ClusterTree`s, where `eta::Float64` is an adjustable parameter.
"""
Base.@kwdef struct StrongAdmissibilityStd <: AbstractAdmissibilityCondition
    eta::Float64=3.0
end

function (adm::StrongAdmissibilityStd)(left_node::ClusterTree, right_node::ClusterTree)
    diam_min = minimum(diameter,(left_node,right_node))
    dist     = distance(left_node,right_node)
    return diam_min < adm.eta*dist
end

"""
    struct WeakAdmissibilityStd <: AbstractAdmissibilityCondition

A `BlockTree` is admissible under this condition if the `distance`
between its `ClusterTree`s is positive.
"""
struct WeakAdmissibilityStd <: AbstractAdmissibilityCondition
end

(adm::WeakAdmissibilityStd)(left_node::ClusterTree, right_node::ClusterTree) = distance(left_node,right_node) > 0


"""
    BlockTree(Xtree::ClusterTree,YTree::ClusterTree,adm)

Main constructor for a `BlockTree` structure.

After intiating the root, calls the recursive `_build_block_tree!` constructor
on the root. The recursive constructor stops if

1. the block is admissible as per `adm`, or
2. either the `col_cluster` or the `row_cluster` is a leaf.

Otherwise the recursive function is called on the `BlockTree`s formed using
the children of `col_cluster` and `row_cluster`.
"""
function BlockTree(row_cluster::ClusterTree, col_cluster::ClusterTree, adm_fun=StrongAdmissibilityStd())
    #build root
    root        = BlockTree(row_cluster,col_cluster,false,(),())
    # recurse
    _build_block_tree!(adm_fun,root)
    return root
end

"""
    _build_block_tree!(adm_fun,current_node)

Recursive constructor for [`BlockTree`](@ref). Should not be called directly.
"""
function _build_block_tree!(adm, current_node)
    if adm(current_node)
        current_node.admissible = true
    else
        current_node.admissible = false
        if !(isleaf(rowcluster(current_node)) || isleaf(colcluster(current_node)))
            row_children       = getchildren(rowcluster(current_node))
            col_children       = getchildren(colcluster(current_node))
            block_children     = [BlockTree(r,c,false,(),current_node) for r in row_children, c in col_children]
            setchildren!(current_node,block_children)
            for child in block_children
                _build_block_tree!(adm,child)
            end
        end
    end
    return current_node
end

function Base.show(io::IO,tree::BlockTree)
    print(io,"BlockTree spanning $(rowrange(tree)) × $(colrange(tree))")
end

function Base.summary(tree::BlockTree)
    print("BlockTree spanning $(rowrange(tree)) × $(colrange(tree))")
    nodes = collect(AbstractTrees.PreOrderDFS(tree))
    @printf "\n\t number of nodes: %i" length(nodes)
    leaves = collect(AbstractTrees.Leaves(tree))
    @printf "\n\t number of leaves: %i" length(leaves)
    points_per_leaf = map(length,leaves)
    @printf "\n\t min number of elements per leaf: %i" minimum(points_per_leaf)
    @printf "\n\t max number of elements per leaf: %i" maximum(points_per_leaf)
    depth_per_leaf = map(depth,leaves)
    @printf "\n\t min depth of leaves: %i" minimum(depth_per_leaf)
    @printf "\n\t max depth of leaves: %i" maximum(depth_per_leaf)
end
