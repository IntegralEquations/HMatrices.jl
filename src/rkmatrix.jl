"""
    mutable struct RkMatrix{T}

Representation of a rank `r` matrix `M` in outer product format `M =
A*adjoint(B)` where `A` has size `m × r` and `B` has size `n × r`.

Two additional fields describe the factors, when this information is available:

- `orthogonality::Symbol`: which factors have mutually orthogonal columns: `:none`,
  `:A`, `:B`, or `:AB` (e.g. a truncated SVD, `A = U*Σ` and `B = V`).
- `truncation_error::Vector{Float64}`: `truncation_error[i]` is an estimate of
  the Frobenius norm of the error made by keeping only the first `i` columns of
  the factors (the last entry estimates the error of `R` itself). It is exact
  (up to rounding) for a truncated SVD of an exact matrix, and an estimate
  otherwise (e.g. the error of a [`PartialACA`](@ref), combined with the one of its
  recompression). An empty vector means that no estimate is available.

Constructing an `RkMatrix` from `A` and `B` only, or assigning `R.A` or `R.B`,
resets these fields to `:none` and `Float64[]`.
"""
mutable struct RkMatrix{T} <: AbstractMatrix{T}
    A::Matrix{T}
    B::Matrix{T}
    orthogonality::Symbol
    truncation_error::Vector{Float64}
    function RkMatrix(
            A::Matrix{T}, B::Matrix{T}, orthogonality::Symbol = :none,
            truncation_error::Vector{Float64} = Float64[],
        ) where {T}
        @assert size(A, 2) == size(B, 2) "second dimension of `A` and `B` must match"
        @assert orthogonality in (:none, :A, :B, :AB) "unknown orthogonality $orthogonality"
        m, r = size(A)
        n = size(B, 1)
        if r * (m + n) >= m * n
            @debug "Inefficient RkMatrix:" size(A) size(B)
        end
        return new{T}(A, B, orthogonality, truncation_error)
    end
end

# assigning a factor invalidates what is known about the factors
function Base.setproperty!(R::RkMatrix, s::Symbol, x)
    setfield!(R, s, convert(fieldtype(typeof(R), s), x))
    if s === :A || s === :B
        setfield!(R, :orthogonality, :none)
        setfield!(R, :truncation_error, Float64[])
    end
    return R
end

"""
    _modified_factor!(R::RkMatrix, s::Symbol)

Update what is known about the factors of `R` after the factor `s` (`:A` or
`:B`) was modified in place: the other factor keeps its orthogonality, and the
truncation errors are no longer known.
"""
function _modified_factor!(R::RkMatrix, s::Symbol)
    o = R.orthogonality
    other = s === :A ? :B : :A
    R.orthogonality = (o === :AB || o === other) ? other : :none
    R.truncation_error = Float64[]
    return R
end

"""
    _svd_truncation_error(S, r, prior = 0.0)

The truncation errors (in the Frobenius norm) of a truncated SVD with singular values
`S`, keeping `r` of them, of a matrix which was itself known up to an error `prior`.
"""
function _svd_truncation_error(S, r, prior = 0.0)
    tail = sqrt(sum(abs2, view(S, (r + 1):length(S)); init = zero(real(eltype(S)))))
    err = Vector{Float64}(undef, r)
    acc = abs2(tail) + abs2(prior)
    for i in r:-1:1
        err[i] = sqrt(acc)
        acc += abs2(S[i])
    end
    return err
end

"""
    _error_estimate(R::RkMatrix)

The estimated error of `R` itself (zero if unknown).
"""
_error_estimate(R::RkMatrix) = isempty(R.truncation_error) ? 0.0 : last(R.truncation_error)

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
    getcol!(col, R, j)

Fill `col` with column `j` of the low-rank matrix `R`.
"""
function getcol!(col, R::AdjRkOrRk, j::Int)
    U, V = _rkfactors(R)
    return mul!(col, U, view(adjoint(V), :, j))
end

Base.copy(R::RkMatrix) = RkMatrix(copy(R.A), copy(R.B), R.orthogonality, copy(R.truncation_error))

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
