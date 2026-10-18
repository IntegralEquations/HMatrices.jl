"""
    mutable struct RkMatrix{T}

Representation of a rank `r` matrix `M` in outer product format `M =
A*adjoint(B)` where `A` has size `m × r` and `B` has size `n × r`.
"""
mutable struct RkMatrix{T} <: AbstractMatrix{T}
    A::Matrix{T}
    B::Matrix{T}
    function RkMatrix(A::Matrix{T}, B::Matrix{T}) where {T}
        @assert size(A, 2) == size(B, 2) "second dimension of `A` and `B` must match"
        m, r = size(A)
        n = size(B, 1)
        if r * (m + n) >= m * n
            @debug "Inefficient RkMatrix:" size(A) size(B)
        end
        return new{T}(A, B)
    end
end

function Base.show(io::IO, ::MIME"text/plain", R::RkMatrix)
    return print(
        io,
        size(R, 1),
        "×",
        size(R, 2),
        " RkMatrix{",
        eltype(R),
        "}",
        " of rank ",
        rank(R),
    )
end

function Base.getindex(::RkMatrix, args...)
    msg = """method `getindex(::RkMatrix,args...)` has been disabled to avoid
    performance pitfalls. Unless you made an explicit call to `getindex`, this
    error usually means that a linear algebra routine involving an
    `RkMatrix` has fallen back to a generic implementation."""
    return error(msg)
end

const AdjRk = Adjoint{<:Any, <:RkMatrix}
const AdjRkOrRk = Union{AdjRk, RkMatrix}

"""
    _rkfactors(R)

The factors `(U, V)` of a low-rank matrix `R = U*adjoint(V)`.
"""
_rkfactors(R::RkMatrix) = (R.A, R.B)
_rkfactors(adjR::AdjRk) = (parent(adjR).B, parent(adjR).A)

"""
    getcol!(col, R, j[, Val(add)])

Fill `col` with column `j` of the low-rank matrix `R`, or add it to `col` if `add` is
`true`.
"""
function getcol!(col, R::AdjRkOrRk, j::Int, ::Val{T} = Val(false)) where {T}
    # NOTE: using a `Val` argument to dispatch on the type of `T` is important
    # for performance
    U, V = _rkfactors(R)
    return if T
        mul!(col, U, view(adjoint(V), :, j), true, true)
    else
        mul!(col, U, view(adjoint(V), :, j))
    end
end

Base.copy(R::RkMatrix) = RkMatrix(copy(R.A), copy(R.B))

Base.size(rmat::RkMatrix) = (size(rmat.A, 1), size(rmat.B, 1))
function Base.isapprox(rmat::RkMatrix, B::AbstractArray, args...; kwargs...)
    return isapprox(Matrix(rmat), B, args...; kwargs...)
end

LinearAlgebra.rank(M::RkMatrix) = size(M.A, 2)

function Base.Matrix(R::AdjRkOrRk)
    U, V = _rkfactors(R)
    return U * adjoint(V)
end
function Base.Matrix(R::RkMatrix{<:SMatrix})
    # collect must be used when we have a matrix of `SMatrix` because of this issue:
    # https://github.com/JuliaArrays/StaticArrays.jl/issues/966#issuecomment-943679214
    return R.A * collect(adjoint(R.B))
end

# FIXME: to support scalar and tensorial problems, we currently allow for T
# to be something other than a plain number. If that is the case, we
# implement a (slow) multiplication algorithm by hand to circumvent a
# problem in LinearAlgebra for the generic multiplication mul!(C,A,B,a,b) when
# C and B are a vectors of static matrices, and A is a matrix of static
# matrices. Should eventually be removed.
function LinearAlgebra.mul!(
        C::AbstractVector,
        Rk::RkMatrix{T},
        F::AbstractVector,
        a::Number,
        b::Number,
    ) where {T <: SMatrix}
    m, n = size(Rk)
    r = rank(Rk)
    tmp = adjoint(Rk.B) * F
    rmul!(C, b)
    for k in 1:r
        tmp[k] *= a
        for i in 1:m
            C[i] += Rk.A[i, k] * tmp[k]
        end
    end
    return C
end

# scalar multiplication
Base.:*(a::Number, R::RkMatrix) = (A = a * R.A; B = copy(R.B); RkMatrix(A, B))
