"""
    hmul!(C::HMatrix,A::HMatrix,B::HMatrix,a,b,compressor)

Similar to `mul!` : compute `C <-- A*B*a + B*b`, where `A,B,C` are
hierarchical matrices and `compressor` is a function/functor used in the
intermediate stages of the multiplication to avoid growring the rank of
admissible blocks after addition is performed.
"""
function hmul!(C::HMatrix, A::HMatrix, B::HMatrix, a, b, compressor)
    b == true || rmul!(C, b)
    @timeit_debug "constructing plan" begin
        plan = plan_hmul(C, A, B, a, 1)
    end
    @timeit_debug "executing plan" begin
        execute!(plan, compressor)
    end
    return C
end

# disable `mul` of hierarchial matrices
function mul!(C::HMatrix, A::HMatrix, B::HMatrix, a::Number, b::Number)
    msg = "use `hmul` to multiply hierarchical matrices"
    return error(msg)
end

"""
    struct HMulNode{S,T}

Tree data structure representing the following computation:
```
    C <-- C + a * ∑ᵢ Aᵢ * Bᵢ
```
where `C = target(node)`, `Aᵢ,Bᵢ` are pairs stored in `sources(node)`, and `a`
is stored in the `multiplier` field.

This structure is used to group the operations required when multiplying
hierarchical matrices so that they can later be executed in a way that minimizes
recompression of intermediate computations.
"""
mutable struct HMulNode{T}
    target::T
    children::Matrix{HMulNode{T}}
    sources::Vector{Tuple{T,T}}
    multiplier::Float64
end

function HMulNode(C::HMatrix, a::Number)
    T = typeof(C)
    chdC = children(C)
    m, n = size(chdC)
    return HMulNode{T}(C, Matrix{HMulNode{T}}(undef, m, n), T[], a)
end

function build_HMulNode_structure(C::HMatrix, a)
    node = HMulNode(C, a)
    chdC = children(C)
    m, n = size(chdC)
    for i in 1:m
        for j in 1:n
            child = build_HMulNode_structure(chdC[i, j], a)
            node.children[i, j] = child
        end
    end
    return node
end

function plan_hmul(C::T, A::T, B::T, a, b) where {T<:HMatrix}
    @assert b == 1
    # root = HMulNode(C)
    root = build_HMulNode_structure(C, a)
    # recurse
    _build_hmul_tree!(root, A, B)
    return root
end

function _build_hmul_tree!(tree::HMulNode, A::HMatrix, B::HMatrix)
    C = tree.target
    if isleaf(A) || isleaf(B) || isleaf(C)
        push!(tree.sources, (A, B))
    else
        ni, nj = blocksize(C)
        _, nk = blocksize(A)
        A_children = children(A)
        B_children = children(B)
        C_children = children(C)
        for i in 1:ni
            for j in 1:nj
                child = tree.children[i, j]
                for k in 1:nk
                    _build_hmul_tree!(child, A_children[i, k], B_children[k, j])
                end
            end
        end
    end
    return tree
end

function Base.show(io::IO, ::MIME"text/plain", tree::HMulNode)
    return print(io,
                 "HMulNode with $(size(children(tree))) children and $(length(sources(tree))) pairs")
end

function Base.show(io::IO, tree::HMulNode)
    return print(io,
                 "HMulNode with $(size(children(tree))) children and $(length(sources(tree))) pairs")
end

function Base.show(io::IO, ::MIME"text/plain", tree::Adjoint{<:Any,<:HMulNode})
    p = parent(tree)
    return print(io,
                 "adjoint HMulNode with $(size(children(p))) children and $(length(sources(p))) pairs")
end

function Base.show(io::IO, tree::Adjoint{<:Any,<:HMulNode})
    p = parent(tree)
    return print(io,
                 "adjoint HMulNode with $(size(children(p))) children and $(length(sources(p))) pairs")
end

# compress the operator L = C + ∑ a*Aᵢ*Bᵢ
function (paca::PartialACA)(plan::HMulNode)
    return _aca_partial(plan, :, :, paca.atol, paca.rank, paca.rtol)
end

# getters
target(node::HMulNode) = node.target
sources(node::HMulNode) = node.sources
multiplier(node::HMulNode) = node.multiplier

# Trees interface
children(node::HMulNode) = node.children
children(node::HMulNode, idxs...) = node.children[idxs]
parent(node::HMulNode) = node.parent
isleaf(node::HMulNode) = isempty(children(node))
isroot(node::HMulNode) = parent(node) === node

# AbstractMatrix interface
Base.size(node::HMulNode) = size(target(node))
Base.eltype(node::HMulNode) = eltype(target(node))

Base.getindex(node::HMulNode, ::Colon, j::Int) = getcol(node, j)

function getcol(node::HMulNode, j)
    m, n = size(node)
    T = eltype(node)
    col = zeros(T, m)
    getcol!(col, node, j)
    return col
end

function getcol!(col, node::HMulNode, j)
    a = multiplier(node)
    C = target(node)
    m, n = size(C)
    T = eltype(C)
    ej = zeros(T, n)
    ej[j] = 1
    # compute j-th column of ∑ Aᵢ Bᵢ
    for (A, B) in sources(node)
        m, k = size(A)
        k, n = size(B)
        tmp = zeros(T, k)
        jg = j + offset(B)[2] # global index on hierarchila matrix B
        getcol!(tmp, B, jg)
        _hgemv_recursive!(col, A, tmp, offset(A))
    end
    # multiply by a
    rmul!(col, a)
    # add the j-th column of C if C has data
    # jg  = j + offset(C)[2] # global index on hierarchila matrix B
    # cj  = getcol(C,jg)
    # axpy!(1,cj,col)
    if hasdata(C)
        d = data(C)
        cj = getcol(d, j)
        axpy!(1, cj, col)
    end
    return col
end

adjoint(node::HMulNode) = Adjoint(node)
Base.size(adjnode::Adjoint{<:Any,<:HMulNode}) = reverse(size(adjnode.parent))
children(adjnode::Adjoint{<:Any,<:HMulNode}) = adjoint(children(adjnode.parent))

Base.getindex(adjnode::Adjoint{<:Any,<:HMulNode}, ::Colon, j::Int) = getcol(adjnode, j)

function getcol(adjnode::Adjoint{<:Any,<:HMulNode}, j)
    m, n = size(adjnode)
    T = eltype(adjnode)
    col = zeros(T, m)
    getcol!(col, adjnode, j)
    return col
end

function getcol!(col, adjnode::Adjoint{<:Any,<:HMulNode}, j)
    node = parent(adjnode)
    a = multiplier(node)
    C = target(node)
    T = eltype(C)
    Ct = adjoint(C)
    m, n = size(Ct)
    ej = zeros(T, n)
    ej[j] = 1
    # compute j-th column of ∑ adjoint(Bᵢ)*adjoint(Aᵢ)
    for (A, B) in sources(node)
        At, Bt = adjoint(A), adjoint(B)
        tmp = zeros(T, size(At, 1))
        # _hgemv_recursive!(tmp,At,ej,offset(At))
        jg = j + offset(At)[2] # global index on hierarchila matrix B
        getcol!(tmp, At, jg)
        _hgemv_recursive!(col, Bt, tmp, offset(Bt))
    end
    # multiply by a
    rmul!(col, conj(a))
    # add the j-th column of Ct if it has data
    # jg  = j + offset(Ct)[2] # global index on hierarchila matrix B
    # cj  = getcol(Ct,jg)
    # axpy!(1,cj,col)
    if hasdata(Ct)
        d = data(Ct)
        cj = getcol(d, j)
        axpy!(1, cj, col)
    end
    return col
end

function execute!(node::HMulNode, compressor)
    execute_node!(node, compressor)
    C = target(node)
    flush_to_children!(C)
    # FIXME: using @threads below breaks things when the `RkMatrix` has its own
    # buffer due to a race condition that I do not yet undestand
    for chd in children(node)
        execute!(chd, compressor)
    end
    return node
end

# non-recursive execution
function execute_node!(node::HMulNode, compressor)
    C = target(node)
    isempty(sources(node)) && (return node)
    a = multiplier(node)
    if isleaf(C) && !isadmissible(C)
        d = data(C)
        for (A, B) in sources(node)
            # _mul_leaf!(C,A,B,a,compressor)
            _mul_dense!(d, A, B, a)
        end
    else
        R = compressor(node)
        setdata!(C, R)
    end
    return node
end

function _mul_dense!(C::Matrix, A, B, a)
    Adata = isleaf(A) ? A.data : A
    Bdata = isleaf(B) ? B.data : B
    if Adata isa HMatrix
        if Bdata isa Matrix
            _mul131!(C, Adata, Bdata, a)
        elseif Bdata isa RkMatrix
            _mul132!(C, Adata, Bdata, a)
        end
    elseif Adata isa Matrix
        if Bdata isa Matrix
            _mul111!(C, Adata, Bdata, a)
        elseif Bdata isa RkMatrix
            _mul112!(C, Adata, Bdata, a)
        elseif Bdata isa HMatrix
            _mul113!(C, Adata, Bdata, a)
        end
    elseif Adata isa RkMatrix
        if Bdata isa Matrix
            _mul121!(C, Adata, Bdata, a)
        elseif Bdata isa RkMatrix
            _mul122!(C, Adata, Bdata, a)
        elseif Bdata isa HMatrix
            _mul123!(C, Adata, Bdata, a)
        end
    end
end

"""
    flush_to_children!(H::HMatrix,compressor)

Transfer the blocks `data` to its children. At the end, set `H.data` to `nothing`.
"""
function flush_to_children!(H::HMatrix)
    T = eltype(H)
    isleaf(H) && (return H)
    hasdata(H) || (return H)
    R::RkMatrix{T} = data(H)
    _add_to_children!(H, R)
    setdata!(H, nothing)
    return H
end

function _add_to_children!(H, R::RkMatrix)
    shift = pivot(H) .- 1
    for block in children(H)
        irange = rowrange(block) .- shift[1]
        jrange = colrange(block) .- shift[2]
        bdata = data(block)
        tmp = RkMatrix(R.A[irange, :], R.B[jrange, :])
        if bdata === nothing
            setdata!(block, tmp)
        else
            axpy!(true, tmp, bdata)
        end
    end
end

"""
    flush_to_leaves!(H::HMatrix,compressor)

Transfer the blocks `data` to its leaves. At the end, set `H.data` to `nothing`.

# See also: [`flush_to_children!`](@ref)
"""
function flush_to_leaves!(H::HMatrix)
    T = eltype(H)
    isleaf(H) && (return H)
    hasdata(H) || (return H)
    R::RkMatrix{T} = data(H)
    _add_to_leaves!(H, R)
    setdata!(H, nothing)
    return H
end

function _add_to_leaves!(H, R::RkMatrix)
    shift = pivot(H) .- 1
    for block in AbstractTrees.Leaves(H)
        irange = rowrange(block) .- shift[1]
        jrange = colrange(block) .- shift[2]
        bdata = data(block)
        tmp = RkMatrix(R.A[irange, :], R.B[jrange, :])
        if bdata === nothing
            setdata!(block, tmp)
        else
            axpy!(true, tmp, bdata)
        end
    end
end

function _mul111!(C::Union{Matrix,SubArray,Adjoint}, A::Union{Matrix,SubArray,Adjoint},
                  B::Union{Matrix,SubArray,Adjoint}, a::Number)
    return mul!(C, A, B, a, true)
end

function _mul112!(C::Union{Matrix,SubArray,Adjoint}, M::Union{Matrix,SubArray,Adjoint},
                  R::RkMatrix, a::Number)
    buffer = M * R.A
    _mul111!(C, buffer, R.Bt, a)
    return C
end

function _mul113!(C::Union{Matrix,SubArray,Adjoint}, M::Union{Matrix,SubArray,Adjoint},
                  H::HMatrix, a::Number)
    T = eltype(C)
    if hasdata(H)
        mat = data(H)
        if mat isa Matrix
            _mul111!(C, M, mat, a)
        elseif mat isa RkMatrix
            _mul112!(C, M, mat, a)
        else
            error()
        end
    end
    for child in children(H)
        shift = pivot(H) .- 1
        irange = rowrange(child) .- shift[1]
        jrange = colrange(child) .- shift[2]
        Cview = @views C[:, jrange]
        Mview = @views M[:, irange]
        _mul113!(Cview, Mview, child, a)
    end
    return C
end

function _mul121!(C::Union{Matrix,SubArray,Adjoint}, R::RkMatrix,
                  M::Union{Matrix,SubArray,Adjoint}, a::Number)
    buffer = R.Bt * M
    return _mul111!(C, R.A, buffer, a)
end

function _mul122!(C::Union{Matrix,SubArray,Adjoint}, R::RkMatrix, S::RkMatrix, a::Number)
    if rank(R) < rank(S)
        _mul111!(C, R.A, (R.Bt * S.A) * S.Bt, a)
    else
        _mul111!(C, R.A * (R.Bt * S.A), S.Bt, a)
    end
    return C
end

function _mul123!(C::Union{Matrix,SubArray,Adjoint}, R::RkMatrix, H::HMatrix, a::Number)
    T = promote_type(eltype(R), eltype(H))
    tmp = zeros(T, size(R.Bt, 1), size(H, 2))
    _mul113!(tmp, R.Bt, H, 1)
    _mul111!(C, R.A, tmp, a)
    return C
end

function _mul131!(C::Union{Matrix,SubArray,Adjoint}, H::HMatrix,
                  M::Union{Matrix,SubArray,Adjoint}, a::Number)
    if isleaf(H)
        mat = data(H)
        if mat isa Matrix
            _mul111!(C, mat, M, a)
        elseif mat isa RkMatrix
            _mul121!(C, mat, M, a)
        else
            error()
        end
    end
    for child in children(H)
        shift = pivot(H) .- 1
        irange = rowrange(child) .- shift[1]
        jrange = colrange(child) .- shift[2]
        Cview = view(C, irange, :)
        Mview = view(M, jrange, :)
        _mul131!(Cview, child, Mview, a)
    end
    return C
end

function _mul132!(C::Union{Matrix,SubArray,Adjoint}, H::HMatrix, R::RkMatrix, a::Number)
    T = promote_type(eltype(H), eltype(R))
    buffer = zeros(T, size(H, 1), size(R.A, 2))
    _mul131!(buffer, H, R.A, 1)
    _mul111!(C, buffer, R.Bt, a)
    return C
end

############################################################################################
# Specializations on gemv:
# The routines below provide specialized version of mul!(C,A,B,a,b) when `A` and
# `B` are vectors
############################################################################################

# 1.2.1
function mul!(y::AbstractVector, R::RkMatrix, x::AbstractVector, a::Number, b::Number)
    # tmp = R.Bt*x
    tmp = mul!(R.buffer, adjoint(R.B), x)
    return mul!(y, R.A, tmp, a, b)
end

# 1.2.1
function mul!(y::AbstractVector, adjR::Adjoint{<:Any,<:RkMatrix}, x::AbstractVector,
              a::Number, b::Number)
    R = parent(adjR)
    # tmp = R.At*x
    tmp = mul!(R.buffer, adjoint(R.A), x)
    return mul!(y, R.B, tmp, a, b)
end

# 1.3.1
"""
    mul!(y::AbstractVector,H::HMatrix,x::AbstractVector,a,b[;global_index,threads])

Perform `y <-- H*x*a + y*b` in place.
"""
function mul!(y::AbstractVector, A::HMatrix, x::AbstractVector, a::Number=1, b::Number=0;
              global_index=use_global_index(), threads=use_threads())
    # since the HMatrix represents A = inv(Pr)*H*Pc, where Pr and Pc are row and column
    # permutations, we need first to rewrite C <-- b*C + a*(inv(Pr)*H*Pc)*B as
    # C <-- inv(Pr)*(b*Pr*C + a*H*(Pc*B)). Following this rewrite, the
    # multiplication is performed by first defining B <-- Pc*B, and C <--
    # Pr*C, doing the multiplication with the permuted entries, and then
    # permuting the result  back C <-- inv(Pr)*C at the end.
    ctree = coltree(A)
    rtree = rowtree(A)
    # permute input
    if global_index
        x = x[colperm(A)]
        y = permute!(y, rowperm(A))
        rmul!(x, a) # multiply in place since this is a new copy, so does not mutate exterior x
    elseif a != 1
        x = a * x # new copy of x since we should not mutate the external x in mul!
    end
    iszero(b) ? fill!(y, zero(eltype(y))) : rmul!(y, b)
    # offset in case A is not indexed starting at (1,1); e.g. A is not the root
    # of and HMatrix
    offset = pivot(A) .- 1
    if threads
        # if a partition of the leaves does not already exist, create one. By
        # default a `hilbert_partition` is created
        # TODO: test the various threaded implementations and chose one.
        # Currently there are two main choices:
        # 1. spawn a task per leaf, and let julia scheduler handle the tasks
        # 2. create a static partition of the leaves and try to estimate the
        #    cost, then spawn one task per block of the partition. In this case,
        #    test if the hilbert partition is really faster than col_partition
        #    or row_partition
        #    Right now the hilbert partition is chosen by default without proper
        #    testing.
        @timeit_debug "partitioning leaves" begin
            haskey(CACHED_PARTITIONS, A) ||
                hilbert_partition(A, Threads.nthreads(), _cost_gemv)
            # haskey(CACHED_PARTITIONS,A) || col_partition(A,Threads.nthreads(),_cost_gemv)
            # haskey(CACHED_PARTITIONS,A) || row_partition(A,Threads.nthreads(),_cost_gemv)
        end
        @timeit_debug "threaded multiplication" begin
            p = CACHED_PARTITIONS[A]
            _hgemv_static_partition!(y, x, p.partition, offset)
            # _hgemv_threads!(y, x, p.partition, offset)  # threaded implementation
        end
    else
        @timeit_debug "serial multiplication" begin
            _hgemv_recursive!(y, A, x, offset) # serial implementation
        end
    end
    # permute output
    global_index && invpermute!(y, rowperm(A))
    return y
end

"""
    _hgemv_recursive!(C,A,B,offset)

Internal function used to compute `C[I] <-- C[I] + A*B[J]` where `I =
rowrange(A) - offset[1]` and `J = rowrange(B) - offset[2]`.

The `offset` argument is used on the caller side to signal if the original
hierarchical matrix had a `pivot` other than `(1,1)`.
"""
function _hgemv_recursive!(C::AbstractVector, A::Union{HMatrix,Adjoint{<:Any,<:HMatrix}},
                           B::AbstractVector, offset)
    T = eltype(A)
    if isleaf(A)
        irange = rowrange(A) .- offset[1]
        jrange = colrange(A) .- offset[2]
        d = data(A)
        # FIXME: can we run into this?
        # (https://github.com/JuliaArrays/StaticArrays.jl/issues/966#issuecomment-943679214).
        # C and B are the "global" vectors handled by the caller, so a view
        # is needed.
        mul!(view(C, irange), d, view(B, jrange), 1, 1)
    else
        for block in children(A)
            _hgemv_recursive!(C, block, B, offset)
        end
    end
    return C
end

function _hgemv_threads!(C::AbstractVector, B::AbstractVector, partition, offset)
    nt = Threads.nthreads()
    # make `nt` copies of C and run in parallel
    buffers = [zero(C) for _ in 1:nt]
    @sync for p in partition
        for block in p
            Threads.@spawn begin
                # FIXME: this is probably unsafe since the _hgemv_recursive! may
                # suspend execution and the buffer may be overwritten by another
                # task on the same thread. For now this function is not used, so
                # it is OK, but it should be either fixed or removed.
                buffer = buffers[Threads.threadid()] # local buffer
                _hgemv_recursive!(buffer, block, B, offset)
            end
        end
    end
    # reduce
    for buffer in buffers
        axpy!(1, buffer, C)
    end
    return C
end

function _hgemv_static_partition!(C::AbstractVector, B::AbstractVector, partition, offset)
    # create a lock for the reduction step
    T = eltype(C)
    mutex = ReentrantLock()
    np = length(partition)
    buffers = [zero(C) for _ in 1:np]
    @sync for n in 1:np
        Threads.@spawn begin
            leaves = partition[n]
            Cloc = buffers[n]
            for leaf in leaves
                irange = rowrange(leaf) .- offset[1]
                jrange = colrange(leaf) .- offset[2]
                data = leaf.data
                mul!(view(Cloc, irange), data, view(B, jrange), 1, 1)
            end
            # reduction
            lock(mutex) do
                return axpy!(1, Cloc, C)
            end
        end
    end
    return C
end

function rmul!(R::RkMatrix, b::Number)
    m, n = size(R)
    if m > n
        rmul!(R.B, conj(b))
    else
        rmul!(R.A, b)
    end
    return R
end

function rmul!(H::HMatrix, b::Number)
    b == true && (return H) # short circuit. If inlined, rmul!(H,true) --> no-op
    if hasdata(H)
        rmul!(data(H), b)
    end
    for child in children(H)
        rmul!(child, b)
    end
    return H
end

"""
    _cost_gemv(A::Union{Matrix,SubArray,Adjoint})

A proxy for the computational cost of a matrix/vector product.
"""
function _cost_gemv(R::RkMatrix)
    return rank(R) * sum(size(R))
end
function _cost_gemv(M::Matrix)
    return length(M)
end
function _cost_gemv(H::HMatrix)
    acc = 0.0
    if isleaf(H)
        acc += _cost_gemv(data(H))
    else
        for c in children(H)
            acc += cost_gemv(c)
        end
    end
    return acc
end
