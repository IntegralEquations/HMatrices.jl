"""
    mutable struct ClusterTree{T,S}

Tree structure used to hierarchically sort data of type `Vector{T}` into
containers of type `S`. For example a
`ClusterTree{SVector{3,Float64},HyperRectangle{3,Float64}}` can be used to sort
points in three dimensions into axis aligned hyperrectangles.

A `ClusterTree` node explicitly stores its `parent` and its `children`. The
`data` field references the data at the root level of the tree which was
used in the construction. The data inside the node is given by
`data[index_range]`. In order for the `data` in each node to be contiguous, the
`data` vector is permuted during the construction of the tree. This permutation
is stored in the `perm` field.
"""
mutable struct ClusterTree{T,S}
    data::Vector{T}
    perm::Vector{Int}
    container::S
    index_range::UnitRange{Int}
    children::Maybe{Vector{ClusterTree{T,S}}}
    parent::Maybe{ClusterTree{T,S}}
end

Base.eltype(tree::ClusterTree{T}) where T = T
container_type(tree::ClusterTree{T,S}) where {T,S} = S

# interface to AbstractTrees
AbstractTrees.children(clt::ClusterTree) = getchildren(clt)

# setters and getters
getchildren(clt::ClusterTree) = clt.children
getparent(clt::ClusterTree)   = clt.parent
getdata(clt::ClusterTree)     = clt.data
getperm(clt::ClusterTree)     = clt.perm
setchildren!(clt::ClusterTree,children) = (clt.children = children)
setparent!(clt::ClusterTree,parent)     = (clt.parent   = parent)
setdata!(clt::ClusterTree,data)     = (clt.data = data)
container(clt::ClusterTree) = clt.container

isleaf(clt::ClusterTree) = getchildren(clt)  === ()
isroot(clt::ClusterTree) = getparent(clt) === ()

diameter(node::ClusterTree)                         = diameter(container(node))
distance(node1::ClusterTree,node2::ClusterTree)     = distance(container(node1), container(node2))

dimension(clt::ClusterTree) = container(clt) |> dimension

Base.length(node::ClusterTree) = length(node.index_range)
Base.range(node::ClusterTree)  = node.index_range

"""
    ClusterTree(data,splitter)

Construct a `ClusterTree` from the  given `data` using the splitting strategy
encoded in `splitter`.
"""
function ClusterTree(;data,splitter=CardinalitySplitter(),container=HyperRectangle(data),reorder=true)
    if reorder
        @info "Input data modified upon construction of ClusterTree"
    else
        data = copy(data)
    end
    n_el    = length(data)
    indices   = collect(1:n_el)
    #build the root, then recurse
    root    = ClusterTree(data,indices,container,1:n_el,(),())
    _build_cluster_tree!(root,splitter)
    return root
end

function _build_cluster_tree!(current_node,splitter)
    if should_split(current_node,splitter)
        children          = split!(current_node,splitter)
        setchildren!(current_node,children)
        for child in children
            setparent!(child,current_node)
            _build_cluster_tree!(child,splitter)
        end
    end
end

function Base.show(io::IO,tree::ClusterTree)
    print(io,"ClusterTree with $(length(tree)) elements of type $(eltype(tree))")
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
    struct GeometricSplitter <: AbstractSplitter

Used to split a `ClusterTree` along the largest dimension if `length(tree)>nmax`.
"""
@Base.kwdef struct GeometricSplitter <: AbstractSplitter
    nmax::Int=50
end

should_split(node::ClusterTree,splitter::GeometricSplitter) = length(node) > splitter.nmax

function split!(cluster::ClusterTree,splitter::GeometricSplitter)
    rec         = container(cluster)
    wmax, imax  = findmax(rec.high_corner - rec.low_corner)
    return _binary_split(cluster, imax, rec.low_corner[imax]+wmax/2, false)
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
    rec  = container(cluster)
    wmax, imax          = findmax(rec.high_corner - rec.low_corner)
    return _binary_split(cluster, imax, rec.low_corner[imax]+wmax/2)
end

"""
    struct QuadOctSplitter <: AbstractSplitter

Used to split a `ClusterTree` into `2^N` equally sized children.
"""
@Base.kwdef struct QuadOctSplitter <: AbstractSplitter
    nmax::Int=50
end

should_split(node::ClusterTree,splitter::QuadOctSplitter) = length(node) > splitter.nmax

function split!(cluster::ClusterTree,splitter::QuadOctSplitter)
    d    = dimension(cluster)
    clusters = [cluster]
    for i=1:d
        rec  = container(cluster)
        pos = (rec.high_corner[i] + rec.low_corner[i])/2
        nel = length(clusters)#2^(i-1)
        for k=1:nel
            clt = popfirst!(clusters)
            append!(clusters,_binary_split(clt,i,pos, false))
        end
    end
    return clusters
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
    data                = getdata(cluster)
    rec                 = container(cluster)
    index_range         = cluster.index_range
    wmax, imax          = findmax(rec.high_corner - rec.low_corner)
    med                 = median(data[n][imax] for n in index_range) # the median along largest axis `imax`
    return _binary_split(cluster, imax, med)
end

# TODO: implement nested dissection splitter
# """
#     struct NestedDissectionSplitter <: AbstractSplitter
# """
# @Base.kwdef struct NestedDissectionSplitter <: AbstractSplitter
#     nmax::Int=50
# end

# function split!(cluster::ClusterTree,splitter::NestedDissectionSplitter)
#     clt1, clt2 = split(cluster,GeometricMinimalSplitter(splitter.nmax))
#     S          = getconnectivity(getdata(cluster))
#     perm       = cluster.perm
#     tmp        = Int[]
#     for i in clt1.index_range
#         for j in clt2.index_range
#             if S[perm[i],perm[j]] != 0
#                 push!(tmp,j)
#                 push!(tmp,i)
#             end
#         end
#     end
#     # sep = Cluster(data,perm,bbox,sep_index_range,(),cluster)
#     return [clt1, clt2, tmp]
# end

"""
    _binary_split(clt::ClusterTree,dir,pos,shrink=true)

Generate two `ClusterTree`s by dividing the `container` of `clt` along `dir` at
`pos`. If `shrink==true`, a minimal container is generated for the children clusters.
"""
function _binary_split(cluster::ClusterTree,dir,pos,shrink=true)
    B                   = container_type(cluster)
    perm                = cluster.perm
    rec                 = container(cluster)
    index_range         = cluster.index_range
    data                = getdata(cluster)
    left_rec, right_rec = split(rec, dir, pos)
    perm_idxs           = Vector{Int}(undef,length(cluster))
    npts_left           = 0
    npts_right          = 0
    #sort the points into left and right rectangle
    for i in cluster.index_range
        pt = data[i]
        if pt in left_rec
            npts_left += 1
            perm_idxs[npts_left] = i
        else
            perm_idxs[length(cluster)-npts_right] = i
            npts_right += 1
        end
    end
    perm[index_range]     = perm[perm_idxs]
    data[index_range]     = data[perm_idxs] # reorders the global index set
    left_index_range      = index_range.start:(index_range.start)+npts_left-1
    right_index_range     = (index_range.start+npts_left):index_range.stop
    if shrink
        left_rec   = B(data[left_index_range])
        right_rec  = B(data[right_index_range])
    end
    clt1 = ClusterTree(data, perm, left_rec,  left_index_range,  (), cluster)
    clt2 = ClusterTree(data, perm, right_rec, right_index_range, (), cluster)
    return [clt1, clt2]
end
