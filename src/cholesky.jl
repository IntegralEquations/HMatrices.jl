const HChol = Cholesky{<:Any,<:HermitianHMatrix}

function Base.getproperty(chol::HChol, s::Symbol)
    flag = getfield(chol, :uplo)
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

Hierarhical cholesky facotrization of `M`, using `comp` to generate the compressed
blocks during the multiplication routines.
"""
function LinearAlgebra.cholesky!(M::HermitianHMatrix, compressor; threads = use_threads())
    # perform the cholesky decomposition of M in place
    T = eltype(M)
    nt = Threads.nthreads()
    chn = Channel{ACABuffer{T}}(nt)
    foreach(i -> put!(chn, ACABuffer(T)), 1:nt)
    _cholesky!(M, compressor, threads, chn)
    # wrap the result in the cholesky structure
    return M |> UpperTriangular |> Cholesky
end

"""
    cholesky!(M::HMatrix;atol=0,rank=typemax(Int),rtol=atol>0 ||
    rank<typemax(Int) ? 0 : sqrt(eps(Float64)))

Hierarhical cholesky facotrization of `M`, using the
`PartialACA(;atol,rtol;rank)` compressor.
"""
function LinearAlgebra.cholesky!(
    M::HermitianHMatrix;
    atol = 0,
    rank = typemax(Int),
    rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
    kwargs...,
)
    compressor = PartialACA(atol, rank, rtol)
    return cholesky!(M, compressor; kwargs...)
end

"""
    cholesky(M::HMatrix,args...;kwargs...)

Hierarchical cholesky factorization. See [`cholesky!`](@ref) for the available
options.
"""
LinearAlgebra.cholesky(M::HermitianHMatrix, args...; kwargs...) =
    cholesky!(deepcopy(M), args...; kwargs...)

function _cholesky!(M::HermitianHMatrix, compressor, threads, bufs = nothing)
    if isleaf(M)
        d = data(M)
        @debug "dense cholesky! on $M"
        cholesky!(d, NOPIVOT())
    else
        @assert !hasdata(M)
        chdM = children(M)
        m, n = size(chdM)
        for i in 1:m
            _cholesky!(chdM[i, i], compressor, threads, bufs)
            for j in (i+1):n
                Lᵢᵢ = adjoint(UpperTriangular(chdM[i, i]))
                ldiv!(Lᵢᵢ, chdM[i, j], compressor, bufs)
            end
            for j in (i+1):m
                for k in (i+1):n
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

function LinearAlgebra.ldiv!(A::HChol, y::AbstractVector; global_index = true)
    p = A.factors # underlying data
    ctree = coltree(p)
    rtree = rowtree(p)
    # permute input
    global_index && permute!(y, loc2glob(ctree))
    L, U = A.L, A.U
    # solve cholesky*x = y through:
    # (a) L(z) = y
    # (b) Ux   = z
    ldiv!(L, y)
    ldiv!(U, y)
    global_index && invpermute!(y, loc2glob(rtree))
    return y
end
