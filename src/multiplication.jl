"""
    hmul!(C::HMatrix,A::HMatrix,B::HMatrix,a,b,compressor)

Similar to `mul!` : compute `C <-- A*B*a + C*b`, where `A,B,C` are hierarchical
matrices and `compressor` is a function/functor used in the intermediate stages
of the multiplication to avoid growing the rank of admissible blocks after
addition is performed.
"""
function hmul!(
        C::T,
        A::HTypes,
        B::HTypes,
        a,
        b,
        compressor,
        bufs_ = nothing,
        Cflag = 'N',
    ) where {T <: HMatrix}
    bufs = isnothing(bufs_) ? _buffer_channel(compressor, eltype(C)) : bufs_
    b == true || rmul!(C, b)
    dict = IdDict{T, Vector{Tuple{eltype(children(A)), eltype(children(B))}}}()
    _plan_dict!(dict, C, A, B, Cflag)
    _hmul!(C, compressor, dict, a, nothing, bufs, Cflag)
    return C
end

hmul!(C::HermitianHMatrix, args...) = hmul!(parent(C), args..., C.uplo)

# a channel holding one buffer of `compressor` per thread
function _buffer_channel(compressor, T)
    nt = Threads.nthreads()
    bufs = [allocate_buffer(compressor, T) for _ in 1:nt]
    chn = Channel{eltype(bufs)}(nt)
    foreach(buf -> put!(chn, buf), bufs)
    return chn
end

function _plan_dict!(dict, C::T, A::HTypes, B::HTypes, Cflag) where {T <: HMatrix}
    pairs = get!(dict, C, Tuple{T, T}[])
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
                (Cflag == 'U' && (i > j)) && continue
                (Cflag == 'L' && (j > i)) && continue
                for k in 1:nk
                    _plan_dict!(
                        dict,
                        C_children[i, j],
                        A_children[i, k],
                        B_children[k, j],
                        i == j ? Cflag : 'N',
                    )
                end
            end
        end
    end
    return dict
end

function _hmul!(C::HMatrix, compressor, dict, a, R, bufs, Cflag)
    execute_node!(C, compressor, dict, a, R, bufs)
    shift = offset(C)
    C_children = children(C)
    ni, nj = size(C_children)
    for i in 1:ni
        for j in 1:nj
            (Cflag == 'U' && (i > j)) && continue
            (Cflag == 'L' && (j > i)) && continue
            chd = C_children[i, j]
            irange = rowrange(chd) .- shift[1]
            jrange = colrange(chd) .- shift[2]
            Rp = data(C)
            Rv = hasdata(C) ? RkMatrix(Rp.A[irange, :], Rp.B[jrange, :]) : nothing
            _hmul!(chd, compressor, dict, a, Rv, bufs, i == j ? Cflag : 'N')
        end
    end
    isleaf(C) || (setdata!(C, nothing))
    return C
end

# non-recursive execution
function execute_node!(C::HMatrix, compressor, dict, a, R, bufs)
    T = typeof(C)
    S = eltype(C)
    pairs = get(dict, C, Tuple{T, T}[])
    isnothing(R) && isempty(pairs) && (return C)
    if isleaf(C) && !isadmissible(C)
        d = data(C)::Matrix{S}
        for (A, B) in pairs
            _mul_dense!(d, A, B, a)
        end
        isnothing(R) || mul!(d, R.A, adjoint(R.B), true, true)
    else
        L = MulLinearOp{S}(data(C), R, pairs, a)
        buf = take!(bufs)
        R = compressor(L, axes(L, 1), axes(L, 2), buf)
        put!(bufs, buf)
        setdata!(C, R)
    end
    return C
end

"""
    struct MulLinearOp{T,V,S} <: AbstractMatrix{T}

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
struct MulLinearOp{T, V, S} <: AbstractMatrix{T}
    R::Union{RkMatrix{T}, Nothing}
    P::Union{RkMatrix{T}, Nothing}
    pairs::Vector{V}
    multiplier::S
end

function MulLinearOp{T}(R, P, pairs::Vector{V}, multiplier::S) where {T, V, S}
    return MulLinearOp{T, V, S}(R, P, pairs, multiplier)
end

# AbstractMatrix interface
function Base.size(L::MulLinearOp)
    isnothing(L.R) || (return size(L.R))
    isnothing(L.P) || (return size(L.P))
    isempty(L.pairs) && (return (0, 0))
    A, B = first(L.pairs)
    return (size(A, 1), size(B, 2))
end

function Base.getindex(L::Union{MulLinearOp, Adjoint{<:Any, <:MulLinearOp}}, args...)
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

function getcol!(col, adjL::Adjoint{<:Any, <:MulLinearOp}, j)
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

function getblock!(out, L::Adjoint{<:Any, <:MulLinearOp}, irange, j::Int)
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

# C ← C + a*A*B, where `A` or `B` is a leaf (which is always the case when `C`
# is a dense leaf)
function _mul_dense!(C::Base.Matrix, A, B, a)
    if isleaf(B)
        _mul_hd!(C, A, data(B), a)
    elseif isleaf(A)
        _mul_dh!(C, data(A), B, a)
    else
        error("one of the factors must be a leaf")
    end
    return C
end

# C ← C + a*H*d, with `d` the data of a leaf
_mul_hd!(C, H, d::AdjOrMat, a) = _mul131!(C, H, d, a)
function _mul_hd!(C, H, d::AdjRkOrRk, a)
    U, V = _rkfactors(d)
    tmp = _mul131!(zeros(eltype(C), size(H, 1), size(U, 2)), H, U, 1)
    return mul!(C, tmp, adjoint(V), a, true)
end

# C ← C + a*d*H, with `d` the data of a leaf
_mul_dh!(C, d::AdjOrMat, H, a) = _mul113!(C, d, H, a)
function _mul_dh!(C, d::AdjRkOrRk, H, a)
    U, V = _rkfactors(d)
    tmp = _mul113!(zeros(eltype(C), size(V, 2), size(H, 2)), adjoint(V), H, 1)
    return mul!(C, U, tmp, a, true)
end

function _mul113!(
        C::Union{Matrix, SubArray, Adjoint},
        M::Union{Matrix, SubArray, Adjoint},
        H::HTypes,
        a::Number,
    )
    if hasdata(H)
        mat = data(H)
        if mat isa AdjOrMat
            mul!(C, M, mat, a, true)
        elseif mat isa AdjRkOrRk
            U, V = _rkfactors(mat)
            mul!(C, M * U, adjoint(V), a, true)
        else
            error("unexpected data of type $(typeof(mat))")
        end
    end
    shift = offset(H)
    for child in children(H)
        irange = rowrange(child) .- shift[1]
        jrange = colrange(child) .- shift[2]
        Cview = @views C[:, jrange]
        Mview = @views M[:, irange]
        _mul113!(Cview, Mview, child, a)
    end
    return C
end

function _mul121!(
        C::Union{Matrix, SubArray, Adjoint},
        R::AdjRkOrRk,
        M::Union{Matrix, SubArray, Adjoint},
        a::Number,
    )
    U, V = _rkfactors(R)
    return mul!(C, U, adjoint(V) * M, a, true)
end

function _mul131!(
        C::Union{Matrix, SubArray, Adjoint},
        H::HTypes,
        M::Union{Matrix, SubArray, Adjoint},
        a::Number,
    )
    if isleaf(H)
        mat = data(H)
        if mat isa AdjOrMat
            mul!(C, mat, M, a, true)
        elseif mat isa AdjRkOrRk
            _mul121!(C, mat, M, a)
        else
            error("$(typeof(mat))")
        end
    end
    shift = offset(H)
    for child in children(H)
        irange = rowrange(child) .- shift[1]
        jrange = colrange(child) .- shift[2]
        Cview = view(C, irange, :)
        Mview = view(M, jrange, :)
        _mul131!(Cview, child, Mview, a)
    end
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
        R::AdjRkOrRk,
        x::AbstractVector,
        a::Number,
        b::Number,
    )
    U, V = _rkfactors(R)
    return mul!(y, U, adjoint(V) * x, a, b)
end

# 1.3.1
"""
    mul!(y::AbstractVector,H::HMatrix,x::AbstractVector,a,b[;threads])

Perform `y <-- H*x*a + y*b` in place.
"""
function LinearAlgebra.mul!(
        y::AbstractVector,
        A::HTypes,
        x::AbstractVector,
        a::Number = 1,
        b::Number = 0;
        threads = true,
    )
    # `A` is stored in the ordering of its trees, so `x` and `y` are permuted into it
    # (`x[colperm(A)]` is a copy that `_mul_tree!` may overwrite), and `y` back
    permute!(y, rowperm(A))
    _mul_tree!(y, A, x[colperm(A)], a, b, threads)
    return invpermute!(y, rowperm(A))
end

# the triangular factors of a factorization act in the ordering of the trees
function LinearAlgebra.mul!(
        y::AbstractVector,
        A::HTriangular,
        x::AbstractVector,
        a::Number = 1,
        b::Number = 0;
        threads = true,
    )
    return _mul_tree!(y, A, copy(x), a, b, threads)
end

"""
    _mul_tree!(y, A, x, a, b, threads)

Perform `y <-- A*x*a + y*b` in the ordering of the trees of `A`, overwriting `x`.
"""
function _mul_tree!(y, A, x, a, b, threads)
    rmul!(x, a)
    iszero(b) ? fill!(y, zero(eltype(y))) : rmul!(y, b)
    # offset in case A is not indexed starting at (1,1); e.g. A is not the root
    # of an HMatrix
    if threads
        _hgemv_threads!(y, x, leaves(A), offset(A))  # threaded implementation
    else
        _hgemv_recursive!(y, A, x, offset(A)) # serial implementation
    end
    return y
end

# FIXME: for matrix multiplication, we slice into columns and call the gemv
# routine. This is a somewhat inefficient way of doing things, but it is simple
# enough.
function LinearAlgebra.mul!(
        Y::AbstractMatrix,
        A::HTypes,
        X::AbstractMatrix,
        a::Number = 1,
        b::Number = 0;
        kwargs...,
    )
    size(Y, 2) == size(X, 2) || throw(DimensionMismatch("size(Y,2) != size(X,2)"))
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
function _hgemv_recursive!(C::AbstractVector, A::Union{HTypes, HTriangular}, B::AbstractVector, offset)
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

function _hgemv_threads!(C::AbstractVector, B::AbstractVector, leaves, offset)
    acc = Threads.Atomic{Int}(1)
    lck = ReentrantLock()
    # spawn np workers to assemble the leaves in parallel
    np = Threads.nthreads()
    nl = length(leaves)
    @sync for _ in 1:np
        Threads.@spawn begin
            buf = zero(C)
            while true
                i = Threads.atomic_add!(acc, 1)
                i > nl && break
                leaf = leaves[i]
                irange = rowrange(leaf) .- offset[1]
                jrange = colrange(leaf) .- offset[2]
                mul!(view(buf, irange), data(leaf), view(B, jrange), 1, 1)
            end
            # add the local buffer to the global buffer
            @lock lck axpy!(true, buf, C)
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
