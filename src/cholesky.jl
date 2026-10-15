const HChol = Cholesky{<:Any, <:HermitianHMatrix}

function Base.getproperty(chol::HChol, s::Symbol)
    H = getfield(chol, :factors) # the underlying hierarchical matrix
    if s == :L
        return adjoint(UpperTriangular(H))
    elseif s == :U
        return UpperTriangular(H)
    else
        return getfield(chol, s)
    end
end

function Base.show(io::IO, ::MIME"text/plain", chol::HChol)
    H = getfield(chol, :factors) # the underlying hierarchical matrix
    return println(io, "Cholesky factorization of $H")
end

"""
    cholesky!(M::HMatrix,comp)

Hierarchical Cholesky factorization of `M`, using `comp` to generate the compressed
blocks during the multiplication routines.
"""
function LinearAlgebra.cholesky!(M::HermitianHMatrix, compressor; threads = use_threads())
    # perform the cholesky decomposition of M in place
    _cholesky!(M, compressor, _buffer_channel(compressor, eltype(M)))
    # wrap the result in the cholesky structure
    return M |> UpperTriangular |> Cholesky
end

"""
    cholesky!(M::HMatrix; atol, rank, rtol, threads = use_threads())

Hierarchical Cholesky factorization of `M`, using the
`PartialACA(; atol, rank, rtol)` compressor.
"""
function LinearAlgebra.cholesky!(M::HermitianHMatrix; threads = use_threads(), kwargs...)
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
        cholesky!(d, NOPIVOT())
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
                    hmul!(
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

# solve with the factors `L*U` of an LU or Cholesky factorization
function LinearAlgebra.ldiv!(A::Union{HLU, HChol}, y::AbstractVector; global_index = true)
    p = A.factors # underlying data
    ctree = coltree(p)
    rtree = rowtree(p)
    # permute input
    global_index && permute!(y, loc2glob(rtree))
    L, U = A.L, A.U
    # solve LUx = y through:
    # (a) L(z) = y
    # (b) Ux   = z
    ldiv!(L, y)
    ldiv!(U, y)
    global_index && invpermute!(y, loc2glob(ctree))
    return y
end
