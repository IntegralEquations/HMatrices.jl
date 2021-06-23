"""
    mutable struct ClusterTree{N,T}

Tree structure used to hierarchically sort points in `N` dimensions.

Each node in the tree contains the indices `loc2glob[loc_idxs]`.
"""
mutable struct ClusterTree{N,T}
    points::Vector{SVector{N,T}}
    weights::Vector{T}
    loc_idxs::UnitRange{Int}
    bounding_box::HyperRectangle{N,T}
    loc2glob::Vector{Int}
    glob2loc::Vector{Int}
    children::Maybe{Vector{ClusterTree{N,T}}}
    parent::Maybe{ClusterTree{N,T}}
end

# interface to AbstractTrees
AbstractTrees.children(clt::ClusterTree) = clt.children

isleaf(clt::ClusterTree) = clt.children  === ()
isroot(clt::ClusterTree) = clt.parent    === ()

getchildren(clt::ClusterTree) = clt.children

diameter(node::ClusterTree)                      = diameter(node.bounding_box)
radius(node::ClusterTree)                        = diameter(node)/2
distance(node1::ClusterTree,node2::ClusterTree)  = distance(node1.bounding_box, node2.bounding_box)

dimension(clt::ClusterTree{N}) where {N} = N

Base.length(node::ClusterTree) = length(node.loc_idxs)
Base.range(node::ClusterTree)  = node.loc_idxs

"""
    ClusterTree(data,splitter)

Construct a `ClusterTree` from the  given `data` using the splitting strategy
encoded in `splitter`.
"""
function ClusterTree(points::Vector{SVector{N,T}},splitter;weights=T[]) where {N,T}
    @timeit "ClusterTree construction" begin
        bbox         = HyperRectangle(points)
        n            = length(points)
        loc_idxs     = 1:n
        loc2glob     = collect(loc_idxs)
        glob2loc     = copy(loc2glob)
        children     = ()
        parent       = ()
        #build the root, then recurse
        root         = ClusterTree(points,weights,loc_idxs,bbox,loc2glob,glob2loc,children,parent)
        _build_cluster_tree!(root,splitter)
        root.glob2loc = invperm(root.loc2glob)
    end
    return root
end

function _build_cluster_tree!(current_node,splitter)
    if should_split(current_node,splitter)
        children          = split!(current_node,splitter)
        current_node.children = children
        for child in children
            child.parent = current_node
            _build_cluster_tree!(child,splitter)
        end
    end
end

"""
    abstract type AbstractSplitter

An `AbstractSplitter` is used to split a [`ClusterTree`](@ref). The interface
requires the following methods:
- `should_split(clt,splitter)` : return a `Bool` determining if the
  `ClusterTree` should be further divided
- `split!(clt,splitter)` : perform the splitting of the `ClusterTree` handling
  the necessary data sorting.

See [`GeometricSplitter`](@ref) for an example of an implementation.
"""
abstract type AbstractSplitter end

"""
    should_split(clt::ClusterTree,splitter::AbstractSplitter)

Determine whether or not a `ClusterTree` should be further divided.
"""
function should_split(clt,splitter)
    abstract_method(splitter)
end

"""
    split!(clt::ClusterTree,splitter::AbstractSplitter)

Divide `clt` using the strategy implemented by `splitter`.
"""
function split!(clt,splitter)
    abstract_method(splitter)
end

"""
    struct DyadicSplitter

Used to split an `N` dimensional `ClusterTree` into `2^N` children until at most
`nmax` points are contained in node *or* the depth `dmax` is reached.
"""
Base.@kwdef struct DyadicSplitter <: AbstractSplitter
    nmax::Int=typemax(Int)
    dmax::Int=-1
end

function should_split(node::ClusterTree,splitter::DyadicSplitter)
    length(node) > splitter.nmax || depth(node) < splitter.dmax
end

function split!(cluster::ClusterTree,splitter::DyadicSplitter)
    d        = dimension(cluster)
    clusters = [cluster]
    for i in 1:d
        rec  = cluster.bounding_box
        pos = (rec.high_corner[i] + rec.low_corner[i])/2
        nel = length(clusters) #2^(i-1)
        for _ in 1:nel
            clt = popfirst!(clusters)
            append!(clusters,_binary_split!(clt,i,pos))
        end
    end
    return clusters
end

"""
    struct GeometricSplitter <: AbstractSplitter

Used to split a `ClusterTree` in half along the largest axis.
"""
@Base.kwdef struct GeometricSplitter <: AbstractSplitter
    nmax::Int=50
end

should_split(node::ClusterTree,splitter::GeometricSplitter) = length(node) > splitter.nmax

function split!(cluster::ClusterTree,splitter::GeometricSplitter)
    rec          = cluster.bounding_box
    wmax, imax   = findmax(rec.high_corner - rec.low_corner)
    left_node, right_node = _binary_split!(cluster, imax, rec.low_corner[imax]+wmax/2)
    return [left_node, right_node]
end

"""
    struct GeometricMinimalSplitter <: AbstractSplitter

Like [`GeometricSplitter`](@ref), but shrinks the children's containters.
"""
@Base.kwdef struct GeometricMinimalSplitter <: AbstractSplitter
    nmax::Int=50
end

should_split(node::ClusterTree,splitter::GeometricMinimalSplitter) = length(node) > splitter.nmax

function split!(cluster::ClusterTree,splitter::GeometricMinimalSplitter)
    rec  = cluster.bounding_box
    wmax, imax  = findmax(rec.high_corner - rec.low_corner)
    mid = rec.low_corner[imax]+wmax/2
    predicate = (x) -> x[imax] < mid
    left_node,right_node =  _binary_split!(predicate,cluster)
    return [left_node, right_node]
end

"""
    struct PrincipalComponentSplitter <: AbstractSplitter
"""
@Base.kwdef struct PrincipalComponentSplitter <: AbstractSplitter
    nmax::Int=50
end

should_split(node::ClusterTree,splitter::PrincipalComponentSplitter) = length(node) > splitter.nmax

function split!(cluster::ClusterTree,splitter::PrincipalComponentSplitter)
    pts       = cluster.points
    loc_idxs  = cluster.loc_idxs
    glob_idxs = view(cluster.loc2glob,loc_idxs)
    xc   = centroid(cluster)
    cov  = sum(glob_idxs) do i
        (pts[i] - xc)*transpose(pts[i] - xc)
    end
    v = eigvecs(cov)[:,end]
    predicate = (x) -> dot(x-xc,v) < 0
    left_node, right_node = _binary_split!(predicate,cluster)
    return [left_node, right_node]
end

function centroid(clt::ClusterTree)
    pts       = clt.points
    loc_idxs  = clt.loc_idxs
    glob_idxs = view(clt.loc2glob,loc_idxs)
    w    = clt.weights
    n    = length(loc_idxs)
    M    = isempty(w) ? n : sum(i->w[i],glob_idxs)
    xc   = isempty(w) ? sum(i->pts[i]/M,glob_idxs) : sum(i->w[i]*pts[i]/M,glob_idxs)
    return xc
end

"""
    struct CardinalitySplitter <: AbstractSplitter

Used to split a `ClusterTree` along the largest dimension if
`length(tree)>nmax`. The split is performed so the `data` is evenly distributed
amongst all children.
"""
@Base.kwdef struct CardinalitySplitter <: AbstractSplitter
    nmax::Int=50
end

should_split(node::ClusterTree,splitter::CardinalitySplitter) = length(node) > splitter.nmax

function split!(cluster::ClusterTree,splitter::CardinalitySplitter)
    points     = cluster.points
    loc_idxs   = cluster.loc_idxs
    glob_idxs  = view(cluster.loc2glob,loc_idxs)
    rec        = cluster.bounding_box
    _, imax  = findmax(rec.high_corner - rec.low_corner)
    med         = median(points[i][imax] for i in glob_idxs) # the median along largest axis `imax`
    predicate = (x) -> x[imax] < med
    left_node, right_node = _binary_split!(predicate,cluster)
    return [left_node, right_node]
end

"""
    _binary_split!(node::ClusterTree,dir,pos)
    _binary_split!(f,node::ClusterTree)

Split a `ClusterTree` into two, sorting all points and data in the process.

Passing a `dir` and `pos` arguments splits the `bounding_box` box of `node`
along direction `dir` at position `pos`, then sorts all points into the
resulting  left/right nodes.

If passed a predicate `f`, each point is sorted
according to whether `f(x)` returns `true` (point sorted on
the left node) or `false` (point sorted on the right node). At the end a minimal
`HyperRectangle` containing all left/right points is created.
"""
function _binary_split!(cluster::ClusterTree{N,T},dir::Int,pos::Number) where {N,T}
    points        = cluster.points
    weights       = cluster.weights
    loc_idxs      = cluster.loc_idxs
    glob_idxs     = view(cluster.loc2glob,loc_idxs)
    glob_idxs_new = view(cluster.glob2loc,loc_idxs)
    npts_left  = 0
    npts_right = 0
    rec = cluster.bounding_box
    left_rec, right_rec = split(rec,dir,pos)
    n                   = length(loc_idxs)
    #sort the points into left and right rectangle
    for i in glob_idxs
        pt = points[i]
        if pt in left_rec
            npts_left += 1
            glob_idxs_new[npts_left]    = i
        else  pt # pt in right_rec
            glob_idxs_new[n-npts_right] = i
            npts_right += 1
        end
    end
    @assert npts_left + npts_right == n "points lost during split"
    # update loc2glob map
    copy!(glob_idxs,glob_idxs_new)
    # new ranges for cluster
    left_indices      = loc_idxs.start:(loc_idxs.start)+npts_left-1
    right_indices     = (loc_idxs.start+npts_left):loc_idxs.stop
    # create children
    clt1 = ClusterTree(points,weights,left_indices,  left_rec,  cluster.loc2glob, cluster.glob2loc,(), cluster)
    clt2 = ClusterTree(points,weights,right_indices, right_rec, cluster.loc2glob, cluster.glob2loc,(), cluster)
    return clt1, clt2
end

function _binary_split!(f::Function,cluster::ClusterTree{N,T}) where {N,T}
    points        = cluster.points
    weights       = cluster.weights
    loc_idxs      = cluster.loc_idxs
    glob_idxs     = view(cluster.loc2glob,loc_idxs)
    glob_idxs_new = view(cluster.glob2loc,loc_idxs)
    npts_left  = 0
    npts_right = 0
    xl_left = xl_right = svector(i->typemax(T),N)
    xu_left = xu_right = svector(i->typemin(T),N)
    n          = length(loc_idxs)
    #sort the points into left and right rectangle
    for i in glob_idxs
        pt = points[i]
        if f(pt)
            xl_left = min.(xl_left,pt)
            xu_left = max.(xu_left,pt)
            npts_left += 1
            glob_idxs_new[npts_left]    = i
        else
            xl_right = min.(xl_right,pt)
            xu_right = max.(xu_right,pt)
            glob_idxs_new[n-npts_right] = i
            npts_right += 1
        end
    end
    @assert npts_left + npts_right == n "points lost during split"
    # update loc2glob map
    copy!(glob_idxs,glob_idxs_new)
    # new ranges for cluster
    left_indices      = loc_idxs.start:(loc_idxs.start)+npts_left-1
    right_indices     = (loc_idxs.start+npts_left):loc_idxs.stop
    # compute bounding boxes
    left_rec   = HyperRectangle(xl_left,xu_left)
    right_rec  = HyperRectangle(xl_right,xu_right)
    # create children
    clt1 = ClusterTree(points,weights,left_indices,  left_rec,  cluster.loc2glob, cluster.glob2loc,(), cluster)
    clt2 = ClusterTree(points,weights,right_indices, right_rec, cluster.loc2glob, cluster.glob2loc,(), cluster)
    return clt1, clt2
end

function Base.show(io::IO,tree::ClusterTree{N,T}) where {N,T}
    print(io,"ClusterTree with $(length(tree)) elements of type Point{$N,$T}")
end

function Base.summary(clt::ClusterTree)
    @printf "Cluster tree with %i elements" length(clt)
    nodes = collect(AbstractTrees.PreOrderDFS(clt))
    @printf "\n\t number of nodes: %i" length(nodes)
    leaves = collect(AbstractTrees.Leaves(clt))
    @printf "\n\t number of leaves: %i" length(leaves)
    points_per_leaf = map(length,leaves)
    @printf "\n\t min number of elements per leaf: %i" minimum(points_per_leaf)
    @printf "\n\t max number of elements per leaf: %i" maximum(points_per_leaf)
    depth_per_leaf = map(depth,leaves)
    @printf "\n\t min depth of leaves: %i" minimum(depth_per_leaf)
    @printf "\n\t max depth of leaves: %i" maximum(depth_per_leaf)
end

"""
    struct StrongAdmissibilityStd

A `BlockTree` is admissible under this condition if the minimum of the
`diameter` of its `ClusterTree`s is smaller than `eta` times the `distance`
between the `ClusterTree`s, where `eta::Float64` is an adjustable parameter.
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

A `BlockTree` is admissible under this condition if the `distance`
between its `ClusterTree`s is positive.
"""
struct WeakAdmissibilityStd
end

(adm::WeakAdmissibilityStd)(left_node::ClusterTree, right_node::ClusterTree) = distance(left_node,right_node) > 0
