"""
    hmul!(C::HMatrix,A::HMatrix,B::HMatrix,a,b,compressor)

Similar to `mul!` : compute `C <-- A*B*a + B*b`, where `A,B,C` are hierarchical
matrices and `compressor` is a function/functor used in the intermediate stages
of the multiplication to avoid growring the rank of admissible blocks after
addition is performed.
"""
function hmul!(C::T, A::T, B::T, a, b, compressor, bufs_ = nothing) where {T<:HMatrix}
    bufs = if isnothing(bufs_)
        S = eltype(C)
        chn = Channel{ACABuffer{S}}(Threads.nthreads())
        foreach(i -> put!(chn, ACABuffer(S)), 1:Threads.nthreads())
        chn
    else
        bufs_
    end
    @assert isroot(C) || !hasdata(parent(C))
    b == true || rmul!(C, b)
    dict = IdDict{T,Vector{NTuple{2,T}}}()
    _plan_dict!(dict, C, A, B)
    _hmul!(C, compressor, dict, a, nothing, bufs)
    return C
end

function _plan_dict!(dict, C::T, A::T, B::T) where {T<:HMatrix}
    pairs = get!(dict, C, Tuple{T,T}[])
    if isleaf(A) || isleaf(B) || isleaf(C)
        push!(pairs, (A, B))
    else
        ni, nj = blocksize(C)
        _, nk = blocksize(A)
        A_children = children(A)
        B_children = children(B)
        C_children = children(C)
        for i in 1:ni
            for j in 1:nj
                for k in 1:nk
                    _plan_dict!(dict, C_children[i, j], A_children[i, k], B_children[k, j])
                end
            end
        end
    end
    return dict
end

function _hmul!(C::HMatrix, compressor, dict, a, R, bufs)
    execute_node!(C, compressor, dict, a, R, bufs)
    shift = pivot(C) .- 1
    for chd in children(C)
        irange = rowrange(chd) .- shift[1]
        jrange = colrange(chd) .- shift[2]
        Rp     = data(C)
        Rv     = hasdata(C) ? RkMatrix(Rp.A[irange, :], Rp.B[jrange, :]) : nothing
        # Rv = hasdata(C) ? view(Rp,irange,jrange) : nothing
        _hmul!(chd, compressor, dict, a, Rv, bufs)
    end
    isleaf(C) || (setdata!(C, nothing))
    return C
end

# non-recursive execution
function execute_node!(C::HMatrix, compressor, dict, a, R, bufs)
    T = typeof(C)
    S = eltype(C)
    pairs = get(dict, C, Tuple{T,T}[])
    isnothing(R) && isempty(pairs) && (return C)
    if isleaf(C) && !isadmissible(C)
        d = data(C)::Matrix{S}
        for (A, B) in pairs
            _mul_dense!(d, A, B, a)
        end
        # isnothing(R) || axpy!(true, R, d)
        isnothing(R) || mul!(d, R.A, adjoint(R.B), true, true)
    else
        L = MulLinearOp(data(C), R, pairs, a)
        buf = take!(bufs)
        R = compressor(L, axes(L, 1), axes(L, 2), buf)
        put!(bufs, buf)
        setdata!(C, R)
    end
    return C
end

"""
    struct MulLinearOp{R,T} <: AbstractMatrix{T}

Abstract matrix representing the following linear operator:
```
    L = R + P + a * ∑ᵢ Aᵢ * Bᵢ
```
where `R` and `P` are of type `RkMatrix{T}`, `Aᵢ,Bᵢ` are of type `HMatrix{R,T}`
and `a` is scalar multiplier. Calling `compressor(L)` produces a low-rank
approximation of `L`, where `compressor` is an [`AbstractCompressor`](@ref).

Note: this structure is used to group the operations required when multiplying
hierarchical matrices so that they can later be executed in a way that minimizes
recompression of intermediate computations.
"""
struct MulLinearOp{R,T,S} <: AbstractMatrix{T}
    R::Union{RkMatrix{T},Nothing}
    # P::Union{RkMatrixBlockView{T},Nothing}
    P::Union{RkMatrix{T},Nothing}
    pairs::Vector{NTuple{2,HMatrix{R,T}}}
    multiplier::S
end

# AbstractMatrix interface
function Base.size(L::MulLinearOp)
    isnothing(L.R) || (return size(L.R))
    isnothing(L.P) || (return size(L.P))
    isempty(L.pairs) && (return (0, 0))
    A, B = first(L.pairs)
    return (size(A, 1), size(B, 2))
end

function Base.getindex(L::Union{MulLinearOp,Adjoint{<:Any,<:MulLinearOp}}, args...)
    return error("calling `getindex` of a `MulLinearOp` has been disabled")
end

function getcol!(col, L::MulLinearOp, j)
    fill!(col, zero(eltype(col)))
    m, n = size(L)
    T = eltype(L)
    # compute j-th column of ∑ Aᵢ Bᵢ
    for (A, B) in L.pairs
        m, k = size(A)
        k, n = size(B)
        tmp = zeros(T, k)
        jg = j + offset(B)[2] # global index on hierarchical matrix B
        getcol!(tmp, B, jg)
        _hgemv_recursive!(col, A, tmp, offset(A))
    end
    # multiply the columns by a
    a = L.multiplier
    rmul!(col, a)
    # add R and P (if they exist)
    R = L.R
    if !isnothing(R)
        getcol!(col, R, j, Val(true))
    end
    P = L.P
    if !isnothing(P)
        getcol!(col, P, j, Val(true))
    end
    return col
end

function getblock!(out, L::MulLinearOp, irange, j::Int)
    @assert irange == 1:size(L, 1)
    return getcol!(out, L, j)
end

function getcol!(col, adjL::Adjoint{<:Any,<:MulLinearOp}, j)
    fill!(col, zero(eltype(col)))
    L = parent(adjL)
    T = eltype(L)
    # compute j-th column of ∑ adjoint(Bᵢ)*adjoint(Aᵢ)
    for (A, B) in L.pairs
        At, Bt = adjoint(A), adjoint(B)
        tmp = zeros(T, size(At, 1))
        jg = j + offset(At)[2]
        getcol!(tmp, At, jg)
        _hgemv_recursive!(col, Bt, tmp, offset(Bt))
    end
    # multiply by a
    a = L.multiplier
    rmul!(col, conj(a))
    # add the j-th column of Ct if it has data
    R = L.R
    if !isnothing(R)
        getcol!(col, adjoint(R), j, Val(true))
    end
    P = L.P
    if !isnothing(P)
        getcol!(col, adjoint(P), j, Val(true))
    end
    return col
end

function getblock!(out, L::Adjoint{<:Any,<:MulLinearOp}, irange, j::Int)
    @assert irange == 1:size(L, 1)
    return getcol!(out, L, j)
end

#=
Multiplication when the target is a dense matrix. The numbering system in the following
`_mulxyz` methods use the following convention
1 --> Matrix (dense)
2 --> RkMatrix (sparse)
3 --> HMatrix (hierarchical)
=#

function _mul_dense!(C::Base.Matrix, A, B, a)
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

function _mul111!(C, A, B, a)
    return mul!(C, A, B, a, true)
end

function _mul112!(
    C::Union{Matrix,SubArray,Adjoint},
    M::Union{Matrix,SubArray,Adjoint},
    R::RkMatrix,
    a::Number,
)
    buffer = M * R.A
    _mul111!(C, buffer, R.Bt, a)
    return C
end

function _mul113!(
    C::Union{Matrix,SubArray,Adjoint},
    M::Union{Matrix,SubArray,Adjoint},
    H::HMatrix,
    a::Number,
)
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

function _mul121!(
    C::Union{Matrix,SubArray,Adjoint},
    R::RkMatrix,
    M::Union{Matrix,SubArray,Adjoint},
    a::Number,
)
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

function _mul131!(
    C::Union{Matrix,SubArray,Adjoint},
    H::HMatrix,
    M::Union{Matrix,SubArray,Adjoint},
    a::Number,
)
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
function LinearAlgebra.mul!(
    y::AbstractVector,
    R::RkMatrix,
    x::AbstractVector,
    a::Number,
    b::Number,
)
    tmp = R.Bt * x
    # tmp = mul!(R.buffer, adjoint(R.B), x)
    return mul!(y, R.A, tmp, a, b)
end

# 1.2.1
function LinearAlgebra.mul!(
    y::AbstractVector,
    adjR::Adjoint{<:Any,<:RkMatrix},
    x::AbstractVector,
    a::Number,
    b::Number,
)
    R = parent(adjR)
    tmp = R.At * x
    # tmp = mul!(R.buffer, adjoint(R.A), x)
    return mul!(y, R.B, tmp, a, b)
end

# 1.3.1
"""
    mul!(y::AbstractVector,H::HMatrix,x::AbstractVector,a,b[;global_index,threads])

Perform `y <-- H*x*a + y*b` in place.
"""
function LinearAlgebra.mul!(
    y::AbstractVector,
    A::Union{HMatrix,Adjoint{<:Any,<:HMatrix}},
    x::AbstractVector,
    a::Number = 1,
    b::Number = 0;
    global_index = use_global_index(),
    threads = use_threads(),
)
    # since the HMatrix represents A = inv(Pr)*H*Pc, where Pr and Pc are row and column
    # permutations, we need first to rewrite C <-- b*C + a*(inv(Pr)*H*Pc)*B as
    # C <-- inv(Pr)*(b*Pr*C + a*H*(Pc*B)). Following this rewrite, the
    # multiplication is performed by first defining B <-- Pc*B, and C <--
    # Pr*C, doing the multiplication with the permuted entries, and then
    # permuting the result  back C <-- inv(Pr)*C at the end.
    if global_index
        # permute input
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
        haspartition(A) || (partition!(:hilbert, A))
        _hgemv_static_partition!(y, x, partition_nodes(A), offset)
        # _hgemv_threads!(y, x, nodes(A_part), offset)  # threaded implementation
    else
        _hgemv_recursive!(y, A, x, offset) # serial implementation
    end
    # permute output
    global_index && invpermute!(y, rowperm(A))
    return y
end

# FIXME: for matrix multiplication, we slice into columns and call the gemv
# routine. This is a somewhat inneficient way of doing things, but it is simple
# enough.
function LinearAlgebra.mul!(
    Y::AbstractMatrix,
    A::Union{HMatrix,Adjoint{<:Any,<:HMatrix}},
    X::AbstractMatrix,
    a::Number = 1,
    b::Number = 0;
    kwargs...,
)
    size(Y, 2) == size(X, 2) || Throw(DimensionMismatch("size(Y,2) != size(X,2)"))
    for k in 1:size(Y, 2)
        mul!(view(Y, :, k), A, view(X, :, k), a, b; kwargs...)
    end
    return Y
end

"""
    _hgemv_recursive!(C,A,B,offset)

Internal function used to compute `C[I] <-- C[I] + A*B[J]` where `I =
rowrange(A) - offset[1]` and `J = rowrange(B) - offset[2]`.

The `offset` argument is used on the caller side to signal if the original
hierarchical matrix had a `pivot` other than `(1,1)`.
"""
function _hgemv_recursive!(
    C::AbstractVector,
    A::Union{HMatrix,Adjoint{<:Any,<:HMatrix}},
    B::AbstractVector,
    offset,
)
    if isleaf(A)
        irange = rowrange(A) .- offset[1]
        jrange = colrange(A) .- offset[2]
        d = data(A)
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
    chn = Channel{typeof(C)}(nt)
    foreach(i -> put!(chn, copy(C)), 1:nt)
    @sync for p in partition
        for block in p
            Threads.@spawn begin
                buf = take!(chn)
                _hgemv_recursive!(buf, block, B, offset)
                put!(chn, buf)
            end
        end
    end
    # reduce
    close(chn)
    for buf in chn
        axpy!(1, buf, C)
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
                mul!(view(Cloc, irange), data(leaf), view(B, jrange), 1, 1)
            end
            # reduction
            lock(mutex) do
                return axpy!(1, Cloc, C)
            end
        end
    end
    return C
end

function LinearAlgebra.rmul!(R::RkMatrix, b::Number)
    m, n = size(R)
    if m > n
        rmul!(R.B, conj(b))
    else
        rmul!(R.A, b)
    end
    return R
end

function LinearAlgebra.rmul!(H::HMatrix, b::Number)
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
