const HChol = Cholesky{<:Any, <:HermitianHMatrix}

function Base.getproperty(chol::HChol, s::Symbol)
    H = getfield(chol, :factors) # the underlying hierarchical matrix
    if s == :L
        return adjoint(UpperTriangular(H))
    elseif s == :U
        return UpperTriangular(H)
    elseif s == :p
        return rowperm(H)
    else
        return getfield(chol, s)
    end
end

function Base.propertynames(::HChol, private::Bool = false)
    return (:L, :U, :p, (private ? fieldnames(Cholesky) : ())...)
end

function Base.show(io::IO, ::MIME"text/plain", chol::HChol)
    H = getfield(chol, :factors) # the underlying hierarchical matrix
    return println(io, "Cholesky factorization of $H")
end

# solve with the factors `L*U` of an LU or Cholesky factorization, which are in the
# ordering of the trees: `L*U*x[q] = y[p]`
function LinearAlgebra.ldiv!(A::Union{HLU, HChol}, y::AbstractVecOrMat)
    H = A.factors # underlying data
    _permute_rows!(y, rowperm(H))
    ldiv!(A.L, y)
    ldiv!(A.U, y)
    return _invpermute_rows!(y, colperm(H))
end

_permute_rows!(y::AbstractVector, p) = permute!(y, p)
_permute_rows!(Y::AbstractMatrix, p) = (Y .= Y[p, :])
_invpermute_rows!(y::AbstractVector, p) = invpermute!(y, p)
_invpermute_rows!(Y::AbstractMatrix, p) = (Y[p, :] = copy(Y); Y)
