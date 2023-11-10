using Distributed

function typeof_future(r::Future)
    if isready(r)
        pid = r.where
        f = @spawnat pid typeof(fetch(r))
        T = fetch(f)
        return T
    else
        error("future is not ready")
        return nothing
    end
end

"""
    struct RemoteHMatrix{S,T}

A light wrapper for a `Future` storing an `HMatrix`.
"""
struct RemoteHMatrix{S,T}
    future::Future
end

Base.fetch(r::RemoteHMatrix) = fetch(r.future)

function Base.getindex(r::RemoteHMatrix, i::Int, j::Int)
    pid = r.future.where
    @fetchfrom pid getindex(fetch(r), i, j)
end

"""
    mutable struct DHMatrix{R,T} <: AbstractHMatrix{T}

Concrete type representing a hierarchical matrix with data distributed amongst
various workers. Its structure is very similar to `HMatrix`, except that the
leaves store a [`RemoteHMatrix`](@ref) object.

The `data` on the leaves of a `DHMatrix` may live on a different worker, so
calling `fetch` on them should be avoided whenever possible.
"""
mutable struct DHMatrix{R,T} <: AbstractHMatrix{T}
    rowtree::R
    coltree::R
    # admissible:: Bool --> false
    data::Union{RemoteHMatrix{R,T},Nothing}
    children::Matrix{DHMatrix{R,T}}
    parent::DHMatrix{R,T}
    # inner constructor which handles `nothing` fields.
    function DHMatrix{R,T}(rowtree, coltree, data, children, parent) where {R,T}
        dhmat = new{R,T}(rowtree, coltree, data)
        dhmat.children = isnothing(children) ? Matrix{DHMatrix{R,T}}(undef, 0, 0) : children
        dhmat.parent = isnothing(parent) ? dhmat : parent
        return dhmat
    end
end

"""
    DHMatrix{T}(rowtree,coltree;partition_strategy=:distribute_columns)

Construct the block structure of a distributed hierarchical matrix covering
`rowtree` and `coltree`. Returns a `DHMatrix` with leaves that are empty.

The `partition_strategy` keyword argument determines how to partition the blocks
for distributed computing. Currently, the only available options is
`distribute_columns`, which will partition the columns of the underlying matrix
into `floor(log2(nw))` parts, where `nw` is the number of workers available.
"""
function DHMatrix{T}(
    rowtree::R,
    coltree::R;
    partition_strategy = :distribute_columns,
) where {R,T}
    #build root
    root = DHMatrix{R,T}(rowtree, coltree, nothing, nothing, nothing)
    # depending on the partition strategy, dispatch to appropriate (recursive)
    # method
    if partition_strategy == :distribute_columns
        nw = nworkers()
        dmax = floor(Int64, log2(nw))
        _build_block_structure_distribute_cols!(root, dmax)
    else
        error("unrecognized partition strategy")
    end
    return root
end

function _build_block_structure_distribute_cols!(
    current_node::DHMatrix{R,T},
    dmax,
) where {R,T}
    if Trees.depth(current_node) == dmax
        return current_node
    else
        X = rowtree(current_node)
        Y = coltree(current_node)
        (isleaf(X) || isleaf(Y)) && error("you should split the tree further")
        # do not recurse on row, only on columns
        row_children = [X]
        col_children = Y.children
        children = [
            DHMatrix{R,T}(r, c, nothing, nothing, current_node) for r in row_children,
            c in col_children
        ]
        current_node.children = children
        for child in children
            _build_block_structure_distribute_cols!(child, dmax)
        end
        return current_node
    end
end

Base.size(H::DHMatrix) = length(rowrange(H)), length(colrange(H))

function Base.show(io::IO, ::MIME"text/plain", hmat::DHMatrix)
    # isclean(hmat) || return print(io,"Dirty DHMatrix")
    println(
        io,
        "Distributed HMatrix of $(eltype(hmat)) with range $(rowrange(hmat)) × $(colrange(hmat))",
    )
    nodes = collect(AbstractTrees.PreOrderDFS(hmat))
    println(io, "\t number of nodes in tree: $(length(nodes))")
    leaves = collect(AbstractTrees.Leaves(hmat))
    @printf(io, "\t number of leaves: %i\n", length(leaves))
    for (i, leaf) in enumerate(leaves)
        r = leaf.data.future
        pid = r.where
        irange, jrange = @fetchfrom pid rowrange(fetch(r)), colrange(fetch(r))
        println("\t\t leaf $i on process $pid spanning $irange × $jrange")
    end
end

"""
    _assemble_hmat_distributed(K,rtree,ctree;adm=StrongAdmissibilityStd(),comp=PartialACA();global_index=true,threads=false)

Internal methods called **after** the `DHMatrix` structure has been initialized
in order to construct the `HMatrix` on each of the leaves of the `DHMatrix`.
"""
function _assemble_hmat_distributed(
    K,
    rtree,
    ctree;
    adm = StrongAdmissibilityStd(),
    comp = PartialACA(),
    global_index = use_global_index(),
    threads = use_threads(),
)
    #
    R = typeof(rtree)
    T = eltype(K)
    wids = workers()
    root = DHMatrix{T}(rtree, ctree; partition_strategy = :distribute_columns)
    leaves = collect(AbstractTrees.Leaves(root))
    @info "Assembling distributed HMatrix on $(length(leaves)) processes"
    @sync for (k, leaf) in enumerate(leaves)
        pid = wids[k] # id of k-th worker
        r = @spawnat pid assemble_hmatrix(
            K,
            rowtree(leaf),
            coltree(leaf);
            adm,
            comp,
            global_index,
            threads,
            distributed = false,
        )
        leaf.data = RemoteHMatrix{R,T}(r)
    end
    return root
end

function LinearAlgebra.mul!(
    y::AbstractVector,
    A::DHMatrix,
    x::AbstractVector,
    a::Number,
    b::Number;
    global_index = use_global_index(),
    threads = use_threads(),
)
    # since the HMatrix represents A = Pr*H*Pc, where Pr and Pc are row and column
    # permutations, we need first to rewrite C <-- b*C + a*(Pc*H*Pb)*B as
    # C <-- Pr*(b*inv(Pr)*C + a*H*(Pc*B)). Following this rewrite, the
    # multiplication is performed by first defining B <-- Pc*B, and C <--
    # inv(Pr)*C, doing the multiplication with the permuted entries, and then
    # permuting the result  C <-- Pr*C at the end.
    ctree = A.coltree
    rtree = A.rowtree
    # permute input
    if global_index
        x = x[ctree.loc2glob]
        y = permute!(y, rtree.loc2glob)
        rmul!(x, a) # multiply in place since this is a new copy, so does not mutate exterior x
    else
        x = a * x # new copy of x
    end
    rmul!(y, b)
    leaves = filter_tree(x -> Trees.isleaf(x), A)
    nb = length(leaves)
    acc = Vector{Future}(undef, nb)
    @sync for i in 1:nb
        r = leaves[i].data.future
        pid = r.where
        jrange = @fetchfrom pid colrange(fetch(r))
        T, n = eltype(y), length(y)
        acc[i] = @spawnat pid mul!(
            zeros(T, n),
            fetch(r),
            view(x, jrange),
            1,
            0;
            global_index = false,
            threads,
        )
    end
    # reduction stage: fetch everybody to worker 1 and sum them up
    for yi in acc
        axpy!(1, fetch(yi), y)
    end
    # permute output
    global_index && invpermute!(y, loc2glob(rtree))
    return y
end

function isclean(H::DHMatrix)
    for node in AbstractTrees.PreOrderDFS(H)
        if isleaf(node)
            if !hasdata(node)
                @warn "leaf node without data found"
                return false
            end
            if !isclean(data(node))
                @warn "dirty `HMatrix` leaf found"
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
