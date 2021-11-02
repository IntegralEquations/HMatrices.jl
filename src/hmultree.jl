"""
    struct HMulNode{S,T} <: AbstractMatrix{T}

Tree data structure representing the following computation:

```
    C <-- C + a * ∑ᵢ Aᵢ * Bᵢ
```

where `C = target(node)`, and `Aᵢ,Bᵢ` are pairs stored in `sources(node)`.

This structure is used to group the operations required when multiplying
hierarchical matrices so that they can later be executed in a way that minimizes
recompression of intermediate computations.
"""
mutable struct HMulNode{T} <: AbstractTree
    target::T
    children::Matrix{HMulNode{T}}
    sources::Vector{Tuple{T,T}}
    multiplier::Float64
end

# getters
target(node::HMulNode) = node.target
sources(node::HMulNode) = node.sources
multiplier(node::HMulNode) = node.multiplier

# Trees interface
Trees.children(node::HMulNode) = node.children
Trees.children(node::HMulNode,idxs...) = node.children[idxs]
Trees.parent(node::HMulNode)   = node.parent
Trees.isleaf(node::HMulNode)   = isempty(children(node))
Trees.isroot(node::HMulNode)   = parent(node) === node

# AbstractMatrix interface
Base.size(node::HMulNode) = size(target(node))
Base.eltype(node::HMulNode) = eltype(target(node))

Base.getindex(node::HMulNode,::Colon,j::Int) = getcol(node,j)
function getcol(node::HMulNode,j)
    m,n = size(node)
    T   = eltype(node)
    col = zeros(T,m)
    getcol!(col,node,j)
    return col
end
function getcol!(col,node::HMulNode,j)
    a = multiplier(node)
    C = target(node)
    m,n = size(C)
    T  = eltype(C)
    ej = zeros(T,n)
    ej[j] = 1
    # compute j-th column of ∑ Aᵢ Bᵢ
    for (A,B) in sources(node)
        m,k = size(A)
        k,n = size(B)
        tmp = zeros(T,k)
        jg   = j + offset(B)[2] # global index on hierarchila matrix B
        getcol!(tmp,B,jg)
        _hgemv_recursive!(col,A,tmp,offset(A))
    end
    # multiply by a
    rmul!(col,a)
    # add the j-th column of C if C has data
    # jg  = j + offset(C)[2] # global index on hierarchila matrix B
    # cj  = getcol(C,jg)
    # axpy!(1,cj,col)
    if hasdata(C)
        d  = data(C)
        cj = getcol(d,j)
        axpy!(1,cj,col)
    end
    return col
end

# the adjoint represents the computation adjoint(C) + conj(a) * ∑ adjoint(Bᵢ)*adjoint(Aᵢ)
LinearAlgebra.adjoint(node::HMulNode) = Adjoint(node)
Base.size(adjnode::Adjoint{<:Any,<:HMulNode}) = reverse(size(adjnode.parent))
# hasdata(adjH::Adjoint{<:Any,<:HMatrix}) = hasdata(adjH.parent)
# data(adjH::Adjoint{<:Any,<:HMatrix}) = adjoint(data(adjH.parent))
Trees.children(adjnode::Adjoint{<:Any,<:HMulNode}) = adjoint(children(adjnode.parent))
# pivot(adjH::Adjoint{<:Any,<:HMatrix}) = reverse(pivot(adjH.parent))
# rowrange(adjH::Adjoint{<:Any,<:HMatrix}) = colrange(adjH.parent)
# colrange(adjH::Adjoint{<:Any,<:HMatrix}) = rowrange(adjH.parent)
# Trees.isleaf(adjH::Adjoint{<:Any,<:HMatrix}) = isleaf(adjH.parent)


Base.getindex(adjnode::Adjoint{<:Any,<:HMulNode},::Colon,j::Int) = getcol(adjnode,j)
function getcol(adjnode::Adjoint{<:Any,<:HMulNode},j)
    m,n = size(adjnode)
    T   = eltype(adjnode)
    col = zeros(T,m)
    getcol!(col,adjnode,j)
    return col
end
function getcol!(col,adjnode::Adjoint{<:Any,<:HMulNode},j)
    node  = parent(adjnode)
    a     = multiplier(node)
    C     = target(node)
    T     = eltype(C)
    Ct    = adjoint(C)
    m,n   = size(Ct)
    ej    = zeros(T,n)
    ej[j] = 1
    # compute j-th column of ∑ adjoint(Bᵢ)*adjoint(Aᵢ)
    for (A,B) in sources(node)
        At,Bt = adjoint(A), adjoint(B)
        tmp = zeros(T,size(At,1))
        # _hgemv_recursive!(tmp,At,ej,offset(At))
        jg  = j + offset(At)[2] # global index on hierarchila matrix B
        getcol!(tmp,At,jg)
        _hgemv_recursive!(col,Bt,tmp,offset(Bt))
    end
    # multiply by a
    rmul!(col,conj(a))
    # add the j-th column of Ct if it has data
    # jg  = j + offset(Ct)[2] # global index on hierarchila matrix B
    # cj  = getcol(Ct,jg)
    # axpy!(1,cj,col)
    if hasdata(Ct)
        d  = data(Ct)
        cj = getcol(d,j)
        axpy!(1,cj,col)
    end
    return col
end

function execute!(node::HMulNode,compressor)
    execute_node!(node,compressor)
    C = target(node)
    flush_to_children!(C,identity)
    for chd in children(node)
        execute!(chd,compressor)
    end
    return node
end

# non-recursive execution
function execute_node!(node,compressor)
    C = target(node)
    isempty(sources(node)) && (return node)
    a = multiplier(node)
    if isleaf(C) && !isadmissible(C)
        for (A,B) in sources(node)
            _mul_leaf!(C,A,B,a,compressor)
        end
    else
        R = compressor(node)
        setdata!(C,R)
    end
    return node
end

"""
    flush_to_children!(H::HMatrix,compressor)

Transfer the blocks `data` to its children. At the end, set `H.data` to `nothing`.
"""
function flush_to_children!(H::HMatrix,compressor)
    T = eltype(H)
    isleaf(H)  && (return H)
    hasdata(H) || (return H)
    R::RkMatrix{T}   = data(H)
    _add_to_children!(H,R,compressor)
    setdata!(H,nothing)
    return H
end

function _add_to_children!(H,R::RkMatrix,compressor)
    shift = pivot(H) .- 1
    for block in children(H)
        irange   = rowrange(block) .- shift[1]
        jrange   = colrange(block) .- shift[2]
        bdata    = data(block)
        tmp      = RkMatrix(R.A[irange,:],R.B[jrange,:])
        if bdata === nothing
            setdata!(block,tmp)
        else
            axpy!(true,tmp,bdata)
            bdata isa RkMatrix && compress!(bdata,compressor)
        end
    end
end

"""
    transfer_to_leaves(H::HMatrix)

Replace the data on the leaves with the data stored in `H`.
"""
function transfer_to_leaves!(H::HMatrix)
    @assert hasdata(H)
    isleaf(H) && (return H)
    R = data(H)
    _transfer_to_leaves!(H,R)
    setdata!(H,nothing)
    return H
end

function _transfer_to_leaves!(H::HMatrix,R::RkMatrix)
    shift = pivot(H) .- 1
    for block in Leaves(H)
        irange     = rowrange(block) .- shift[1]
        jrange     = colrange(block) .- shift[2]
        bdata      = data(block)
        if isadmissible(block)
            d = RkMatrix(R.A[irange,:],R.B[jrange,:])
            setdata!(block,d)
        else
            d = @views R.A[irange,:]*R.Bt[:,jrange]
            setdata!(block,d)
        end
    end
    return H
end

function HMulNode(C::HMatrix,a)
    T    = typeof(C)
    chdC = children(C)
    m,n  = size(chdC)
    HMulNode{T}(C,Matrix{HMulNode{T}}(undef,m,n),T[],a)
end

function build_HMulNode_structure(C::HMatrix,a)
    node = HMulNode(C,a)
    chdC = children(C)
    m,n  = size(chdC)
    for i in 1:m
        for j in 1:n
            child = build_HMulNode_structure(chdC[i,j],a)
            node.children[i,j] = child
        end
    end
    node
end

function plan_hmul(C::T,A::T,B::T,a,b) where {T<:HMatrix}
    @assert b == 1
    # root = HMulNode(C)
    root = build_HMulNode_structure(C,a)
    # recurse
    _build_hmul_tree!(root,A,B)
    return root
end

function _build_hmul_tree!(tree::HMulNode,A::HMatrix,B::HMatrix)
    C = tree.target
    if isleaf(A) || isleaf(B) || isleaf(C)
        push!(tree.sources,(A,B))
    else
        ni,nj = blocksize(C)
        _ ,nk = blocksize(A)
        A_children = children(A)
        B_children = children(B)
        C_children = children(C)
        for i=1:ni
            for j=1:nj
                child = tree.children[i,j]
                for k=1:nk
                    _build_hmul_tree!(child,A_children[i,k],B_children[k,j])
                end
            end
        end
    end
    return tree
end

function Base.show(io::IO,::MIME"text/plain",tree::HMulNode)
    print(io,"HMulNode with $(size(children(tree))) children and $(length(sources(tree))) pairs")
end

function Base.show(io::IO,tree::HMulNode)
    print(io,"HMulNode with $(size(children(tree))) children and $(length(sources(tree))) pairs")
end

function Base.show(io::IO,::MIME"text/plain",tree::Adjoint{<:Any,<:HMulNode})
    p = parent(tree)
    print(io,"adjoint HMulNode with $(size(children(p))) children and $(length(sources(p))) pairs")
end

function Base.show(io::IO,tree::Adjoint{<:Any,<:HMulNode})
    p = parent(tree)
    print(io,"adjoint HMulNode with $(size(children(p))) children and $(length(sources(p))) pairs")
end

# compress the operator L = C + ∑ a*Aᵢ*Bᵢ
function (paca::PartialACA)(plan::HMulNode)
    _aca_partial(plan,:,:,paca.atol,paca.rank,paca.rtol)
end

"""
    hmul_with_plan!(C::HMatrix,A::HMatrix,B::HMatrix,a,b,compressor)

Like [`hmul!`](@ref), but builds plan first and then
"""
function hmul_with_plan!(C::HMatrix,A::HMatrix,B::HMatrix,a,b,compressor)
    b == true || rmul!(C,b)
    plan = plan_hmul(C,A,B,a,1)
    execute!(plan,comp)
    return C
end
