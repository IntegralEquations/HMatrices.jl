"""
    struct LeafOperator{T,M,V,S} <: AbstractMatrix{T}

The operator
```
    L = A * B' + α * ∑ᵢ Xᵢ * Yᵢ
```
known through its rows and columns, or its products with blocks of vectors, where `(A,
B)` are low-rank factors (of type `M`), `(Xᵢ, Yᵢ)` are pairs of hierarchical matrices
(of type `V`), and `α` is a scalar. It is the value of an admissible leaf with its
deferred updates, which a compressor compresses once (see `_flush_low_rank_leaf!`):
`compressor(L, axes(L)..., buf)`, through `getblock!` (see [`PartialACA`](@ref)), or
`mul!` (see [`RandomizedSVD`](@ref)).

    LeafOperator{T}(A, B, pairs, α, buffer = T[], rkbuffer = T[])

The scratch vectors `buffer` and `rkbuffer` grow as needed, and can be reused (e.g.
taken from the workspace of a flush).
"""
struct LeafOperator{T, M, V, S} <: AbstractMatrix{T}
    A::M
    B::M
    pairs::Vector{V}
    α::S
    # scratch space for the intermediate vectors of the products `Xᵢ*(Yᵢ*x)`, and for
    # the products with the low-rank leaves of `Xᵢ`
    buffer::Vector{T}
    rkbuffer::Vector{T}
end

function LeafOperator{T}(A::M, B::M, pairs::Vector{V}, α::S, buffer = T[], rkbuffer = T[]) where {T, M, V, S}
    return LeafOperator{T, M, V, S}(A, B, pairs, α, buffer, rkbuffer)
end

Base.size(L::LeafOperator) = (size(L.A, 1), size(L.B, 1))

function Base.getindex(L::Union{LeafOperator, Adjoint{<:Any, <:LeafOperator}}, args...)
    return error("calling `getindex` of a `LeafOperator` has been disabled")
end

"""
    _oriented(L::LeafOperator, ::Val{trans})

The low-rank factors and the coefficient of `L` (`trans = false`), or of its adjoint
`L' = B*A' + conj(α)*∑ᵢ Yᵢ'*Xᵢ'` (`trans = true`); see `_product`.
"""
_oriented(L::LeafOperator, ::Val{false}) = (L.A, L.B, L.α)
_oriented(L::LeafOperator, ::Val{true}) = (L.B, L.A, conj(L.α))

"""
    _product(X, Y, ::Val{trans})

The factors of the product `X*Y` (`trans = false`) or of its adjoint (`trans = true`).
"""
_product(X, Y, ::Val{false}) = (X, Y)
_product(X, Y, ::Val{true}) = (adjoint(Y), adjoint(X))

"""
    _operator_col!(col, L::LeafOperator, trans::Val, j)

The column `j` of `L` (or of its adjoint, see `_oriented`), in `col`.
"""
function _operator_col!(col, L::LeafOperator, trans, j)
    fill!(col, zero(eltype(col)))
    A, B, α = _oriented(L, trans)
    for (X, Y) in L.pairs
        X, Y = _product(X, Y, trans)
        tmp = fill!(_scratch!(L.buffer, size(X, 2)), zero(eltype(L)))
        getcol!(tmp, Y, j + offset(Y)[2])
        _hgemv_recursive!(col, X, tmp, offset(X), L.rkbuffer)
    end
    rmul!(col, α)
    _add_lowrank_col!(col, A, B, j, L.buffer)
    return col
end

"""
    _add_lowrank_col!(col, A, B, j, buf)

`col ← col + (A*B')[:, j]`, with the row `j` of `B` adjoined into the scratch vector
`buf`, so that the product is a matrix-vector product with `A` (BLAS for
numbers, which a view of `adjoint(B)` with complex entries does not reach).
"""
function _add_lowrank_col!(col, A, B, j, buf)
    x = _scratch!(buf, size(B, 2))
    for k in eachindex(x)
        x[k] = adjoint(B[j, k])
    end
    return mul!(col, A, x, true, true)
end

# an entry costs a column
_can_sample(::LeafOperator) = false

function getblock!(out, L::LeafOperator, irange, j::Int)
    @assert irange == 1:size(L, 1)
    return _operator_col!(out, L, Val(false), j)
end
function getblock!(out, L::Adjoint{<:Any, <:LeafOperator}, irange, j::Int)
    @assert irange == 1:size(L, 1)
    return _operator_col!(out, parent(L), Val(true), j)
end
# a whole block (e.g. for `TSVD`), column after column
function getblock!(out::AbstractMatrix, L::LeafOperator, irange, J::AbstractVector{<:Integer})
    @assert irange == 1:size(L, 1)
    for (t, j) in enumerate(J)
        _operator_col!(view(out, :, t), L, Val(false), j)
    end
    return out
end
"""
    _mul!(out, L::LeafOperator, V, trans::Val)

`out = L*V` (or `L'*V`, see `_oriented`), through the products of the hierarchical
matrices with the block of vectors `V`.
"""
function _mul!(out, L::LeafOperator, V, trans)
    fill!(out, zero(eltype(out)))
    A, B, α = _oriented(L, trans)
    for (X, Y) in L.pairs
        X, Y = _product(X, Y, trans)
        tmp = fill!(_scratch!(L.buffer, (size(Y, 1), size(V, 2))), zero(eltype(L)))
        _mul131!(tmp, Y, V, true, L.rkbuffer)
        _mul131!(out, X, tmp, α, L.rkbuffer)
    end
    tmp = mul!(_scratch!(L.buffer, (size(B, 2), size(V, 2))), adjoint(B), V)
    return mul!(out, A, tmp, true, true)
end

LinearAlgebra.mul!(out::Matrix, L::LeafOperator, V::Matrix) = _mul!(out, L, V, Val(false))
LinearAlgebra.mul!(out::Matrix, L::Adjoint{<:Any, <:LeafOperator}, V::Matrix) = _mul!(out, parent(L), V, Val(true))

function (comp::RandomizedSVD)(L::LeafOperator, irange::AbstractRange, jrange::AbstractRange, buf = nothing)
    @assert irange == axes(L, 1) && jrange == axes(L, 2)
    return _randomized_svd(L, comp)
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

function _mul131!(
        C::AbstractVecOrMat,
        H::HTypes,
        M::AbstractVecOrMat,
        a::Number,
        buf = nothing,
    )
    if isleaf(H)
        mat = data(H)
        if mat isa AdjOrMat
            mul!(C, mat, M, a, true)
        elseif mat isa AdjRkOrRk
            _leaf_gemm!(C, mat, M, a, buf)
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
        _mul131!(Cview, child, Mview, a, buf)
    end
    return C
end

"""
    _leaf_gemm!(C, R, M, a, buf)

`C ← C + a*R*M` for a low-rank leaf `R`, with the intermediate product in the scratch
vector `buf` if one is given (and the entries are numbers).
"""
function _leaf_gemm!(C, R, M, a, buf)
    U, V = _rkfactors(R)
    return mul!(C, U, adjoint(V) * M, a, true)
end
function _leaf_gemm!(C, R::Union{RkMatrix{<:Number}, Adjoint{<:Any, <:RkMatrix{<:Number}}}, M, a, buf::Vector)
    U, V = _rkfactors(R)
    tmp = mul!(_scratch!(buf, (size(V, 2), Base.tail(size(M))...)), adjoint(V), M)
    return mul!(C, U, tmp, a, true)
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
    _checkroot(A)
    # `A` is stored in the ordering of its trees, so `x` and `y` are permuted into it,
    # and `y` back; `x` is copied first in case it aliases `y`
    xp = a .* view(x, colperm(A))
    permute!(y, rowperm(A))
    _mul_tree!(y, A, xp, b, threads)
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
    return _mul_tree!(y, A, a .* x, b, threads)
end

function Base.:*(A::HTriangular, X::AbstractMatrix)
    Y = similar(X, promote_type(eltype(A), eltype(X)), size(A, 1), size(X, 2))
    return mul!(Y, A, X)
end

"""
    _mul_tree!(y, A, x, b, threads)

Perform `y <-- A*x + y*b` in the ordering of the trees of `A`.
"""
function _mul_tree!(y, A, x, b, threads)
    iszero(b) ? fill!(y, zero(eltype(y))) : rmul!(y, b)
    # the indices of the leaves are relative to the first row and column of `A`
    if threads
        _hgemv_threads!(y, x, leaves(A), offset(A))  # threaded implementation
    else
        _hgemv_recursive!(y, A, x, offset(A), promote_type(eltype(A), eltype(x))[]) # serial implementation
    end
    return y
end

# FIXME: for matrix multiplication, we slice into columns and call the gemv
# routine. This is a somewhat inefficient way of doing things, but it is simple
# enough.
function LinearAlgebra.mul!(
        Y::AbstractMatrix,
        A::Union{HTypes, HTriangular},
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
    _hgemv_recursive!(C, A, B, offset[, buf])

`C[I] ← C[I] + A*B[J]`, where `I = rowrange(A) .- offset[1]` and `J = colrange(A) .-
offset[2]`: the vectors `B` and `C` are indexed relative to `offset` (e.g. the offset of
the block of which `A` is a descendant). The products with low-rank leaves use the
scratch vector `buf` if one is given (see `_leaf_gemv!`).
"""
function _hgemv_recursive!(C::AbstractVector, A::HTypes, B::AbstractVector, offset, buf = nothing)
    if isleaf(A)
        _leaf_gemv!(C, A, B, offset, buf)
    else
        for block in children(A)
            _hgemv_recursive!(C, block, B, offset, buf)
        end
    end
    return C
end
# the children in the triangle, the diagonal ones being triangles themselves
function _hgemv_recursive!(C::AbstractVector, A::HTriangular, B::AbstractVector, offset, buf = nothing)
    if isleaf(A)
        _leaf_gemv!(C, A, B, offset, buf)
    else
        chd = children(parent(A))
        for j in axes(chd, 2), i in axes(chd, 1)
            if i == j
                _hgemv_recursive!(C, _wraptriangular(chd[i, i], A), B, offset, buf)
            elseif A isa HLowerTriangular ? i > j : i < j
                _hgemv_recursive!(C, chd[i, j], B, offset, buf)
            end
        end
    end
    return C
end
_leaf_gemv!(C, A, B, offset, buf) = _leaf_gemv!(view(C, rowrange(A) .- offset[1]), data(A), view(B, colrange(A) .- offset[2]), buf)

"""
    _leaf_gemv!(y, d, x, buf)

`y ← y + d*x` for the data `d` of a leaf, with the intermediate product of a low-rank
leaf in the scratch vector `buf` if one is given (and the entries are numbers).
"""
_leaf_gemv!(y, d, x, buf) = mul!(y, d, x, 1, 1)
function _leaf_gemv!(y, R::Union{RkMatrix{<:Number}, Adjoint{<:Any, <:RkMatrix{<:Number}}}, x, buf::Vector)
    U, V = _rkfactors(R)
    tmp = mul!(_scratch!(buf, size(V, 2)), adjoint(V), x)
    return mul!(y, U, tmp, 1, 1)
end

"""
    _scratch!(buf::Vector, k)
    _scratch!(buf::Vector, sz::Tuple)

The first `k` entries of `buf` (as a vector), or the first `prod(sz)` entries (as a
matrix of size `sz`), growing `buf` if needed.
"""
_scratch!(buf::Vector, k::Integer) = (length(buf) < k && resize!(buf, k); view(buf, 1:k))
_scratch!(buf::Vector, sz::Tuple) = reshape(_scratch!(buf, prod(sz)), sz)

function _hgemv_threads!(C::AbstractVector, B::AbstractVector, leaves, offset)
    acc = Threads.Atomic{Int}(1)
    lck = ReentrantLock()
    # spawn np workers to assemble the leaves in parallel
    np = Threads.nthreads()
    nl = length(leaves)
    @sync for _ in 1:np
        Threads.@spawn begin
            buf = zero(C)
            scratch = promote_type(eltype(C), eltype(B))[]
            while true
                i = Threads.atomic_add!(acc, 1)
                i > nl && break
                leaf = leaves[i]
                irange = rowrange(leaf) .- offset[1]
                jrange = colrange(leaf) .- offset[2]
                _leaf_gemv!(view(buf, irange), data(leaf), view(B, jrange), scratch)
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
    # scaling a factor keeps its columns orthogonal, and scales the errors
    R.truncation_error .*= abs(b)
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
    hmul!(C::HMatrix, A, B, a, b, compressor; executor = SerialExecutor())

Similar to `mul!` : compute `C <-- A*B*a + C*b`, where `A,B,C` are hierarchical
matrices (`A` and `B` possibly adjoints, `C` possibly a `Hermitian`, of which only
the stored triangle is updated). Every admissible leaf of `C` is compressed once,
with `compressor`, from its data and all the products which reach it (see
[`LazyDriver`](@ref)), the numeric work being run by `executor`.

The rows of `A` and `C`, the columns of `A` and the rows of `B`, and the columns of
`B` and `C` must be clustered in the same way (e.g. built from the same points);
otherwise an `ArgumentError` is thrown.
"""
function hmul!(C::Union{HMatrix, HermitianHMatrix}, A, B, a, b, compressor; kwargs...)
    rowperm(C) == rowperm(A) && colperm(A) == rowperm(B) && colperm(B) == colperm(C) ||
        throw(ArgumentError("the factors of `hmul!` must be clustered consistently with each other and with the result"))
    return _hmul_tree!(C, A, B, a, b, compressor; kwargs...)
end

"""
    _hmul_tree!(C, A, B, a, b, compressor; uplo = 'N', executor = SerialExecutor())

Like [`hmul!`](@ref), in the ordering in which the H-matrices are stored, i.e. as
products of the stored blocks, whatever the points they come from. With `uplo = 'U'`
(or `'L'`), only the upper (or lower) triangle of the diagonal blocks of `C` is updated.
"""
function _hmul_tree!(
        C::HMatrix, A::Union{HMatrix, AdjointHMatrix}, B::Union{HMatrix, AdjointHMatrix}, a, b, compressor;
        uplo = 'N', executor = SerialExecutor(),
    )
    b == true || rmul!(C, b)
    d = LazyDriver(C, A, B, compressor; α = a, uplo, executor)
    _run!(executor) do
        push!(d, C, A, B)
        flush_tree!(d, C)
    end
    return C
end
function _hmul_tree!(C::HermitianHMatrix, A, B, a, b, compressor; kwargs...)
    _hmul_tree!(parent(C), A, B, a, b, compressor; uplo = C.uplo, kwargs...)
    return C
end
function _hmul_tree!(C, A, B, a, b, compressor; kwargs...)
    throw(ArgumentError("the factors of `hmul!` must be `HMatrix`s or their adjoints"))
end
