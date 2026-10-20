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

"""
    cholesky!(M::HMatrix,comp)

Hierarchical Cholesky factorization of `M`, using `comp` to generate the compressed
blocks during the multiplication routines.

`F \\ b` solves in the ordering of the points used to assemble `M`. The factors `F.L`
and `F.U` are stored in the ordering of the cluster tree, with
`F.L * (F.U * x) ≈ Matrix(M)[F.p, F.p] * x`; products and solves with them act in that ordering.

The factorization is currently serial: `threads` is accepted but not used.
"""
function LinearAlgebra.cholesky!(M::HermitianHMatrix, compressor; threads = true)
    # perform the cholesky decomposition of M in place
    _cholesky!(M, compressor, _buffer_channel(compressor, eltype(M)))
    # wrap the result in the cholesky structure
    return M |> UpperTriangular |> Cholesky
end

"""
    cholesky!(M::HMatrix; atol, rank, rtol, threads = true)

Hierarchical Cholesky factorization of `M`, using the
`PartialACA(; atol, rank, rtol)` compressor.
"""
function LinearAlgebra.cholesky!(M::HermitianHMatrix; threads = true, kwargs...)
    return cholesky!(M, PartialACA(; kwargs...); threads)
end

"""
    cholesky(M::HMatrix,args...;kwargs...)

Hierarchical cholesky factorization. See [`cholesky!`](@ref) for the available
options.
"""
function LinearAlgebra.cholesky(M::HermitianHMatrix, args...; kwargs...)
    return cholesky!(deepcopy(M), args...; kwargs...)
end

function _cholesky!(M::HermitianHMatrix, compressor, bufs = nothing)
    if isleaf(M)
        d = data(M)
        @debug "dense cholesky! on $M"
        cholesky!(d, NoPivot())
    else
        @assert !hasdata(M)
        chdM = children(M)
        m, n = size(chdM)
        for i in 1:m
            _cholesky!(chdM[i, i], compressor, bufs)
            for j in (i + 1):n
                Lᵢᵢ = adjoint(UpperTriangular(chdM[i, i]))
                ldiv!(Lᵢᵢ, chdM[i, j], compressor, bufs)
            end
            for j in (i + 1):m
                for k in (i + 1):n
                    _hmul_tree!(
                        chdM[j, k],
                        adjoint(chdM[i, j]),
                        chdM[i, k],
                        -1,
                        1,
                        compressor,
                        bufs,
                    )
                end
            end
        end
    end
    return M
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
