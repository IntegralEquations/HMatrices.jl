const NOPIVOT = VERSION >= v"1.7" ? NoPivot : Val{false}

const HLU = LU{<:Any,<:HMatrix}

function Base.getproperty(LU::HLU, s::Symbol)
    H = getfield(LU, :factors) # the underlying hierarchical matrix
    if s == :L
        return UnitLowerTriangular(H)
    elseif s == :U
        return UpperTriangular(H)
    else
        return getfield(LU, s)
    end
end

function Base.show(io::IO, ::MIME"text/plain", LU::HLU)
    H = getfield(LU, :factors) # the underlying hierarchical matrix
    return println(io, "LU factorization of $H")
end

"""
    lu!(M::HMatrix,comp)

Hierarhical LU facotrization of `M`, using `comp` to generate the compressed
blocks during the multiplication routines.
"""
function LinearAlgebra.lu!(M::HMatrix, compressor; threads = use_threads())
    # perform the lu decomposition of M in place
    T = eltype(M)
    nt = Threads.nthreads()
    chn = Channel{ACABuffer{T}}(nt)
    foreach(i -> put!(chn, ACABuffer(T)), 1:nt)
    _lu!(M, compressor, threads, chn)
    # wrap the result in the LU structure
    return LU(M, LinearAlgebra.BlasInt[], LinearAlgebra.BlasInt(0))
end

"""
    lu!(M::HMatrix;atol=0,rank=typemax(Int),rtol=atol>0 ||
    rank<typemax(Int) ? 0 : sqrt(eps(Float64)))

Hierarhical LU facotrization of `M`, using the `PartialACA(;atol,rtol;rank)` compressor.
"""
function LinearAlgebra.lu!(
    M::HMatrix;
    atol = 0,
    rank = typemax(Int),
    rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
    kwargs...,
)
    compressor = PartialACA(atol, rank, rtol)
    return lu!(M, compressor; kwargs...)
end

"""
    LinearAlgebra.lu(M::HMatrix,args...;kwargs...)

Hierarchical LU factorization. See [`lu!`](@ref) for the available options.
"""
LinearAlgebra.lu(M::HMatrix, args...; kwargs...) = lu!(deepcopy(M), args...; kwargs...)

function _lu!(M::HMatrix, compressor, threads, bufs = nothing)
    if isleaf(M)
        d = data(M)
        @assert d isa Matrix
        lu!(d, NOPIVOT())
    else
        @assert !hasdata(M)
        chdM = children(M)
        m, n = size(chdM)
        for i in 1:m
            _lu!(chdM[i, i], compressor, threads, bufs)
            for j in (i+1):n
                ldiv!(UnitLowerTriangular(chdM[i, i]), chdM[i, j], compressor, bufs)
                rdiv!(chdM[j, i], UpperTriangular(chdM[i, i]), compressor, bufs)
            end
            for j in (i+1):m
                for k in (i+1):n
                    hmul!(chdM[j, k], chdM[j, i], chdM[i, k], -1, 1, compressor, bufs)
                end
            end
        end
    end
    return M
end

function LinearAlgebra.ldiv!(A::LU{<:Any,<:HMatrix}, y::AbstractVector; global_index = true)
    p = A.factors # underlying data
    ctree = coltree(p)
    rtree = rowtree(p)
    # permute input
    global_index && permute!(y, loc2glob(ctree))
    L, U = A.L, A.U
    # solve LUx = y through:
    # (a) L(z) = y
    # (b) Ux   = z
    ldiv!(L, y)
    ldiv!(U, y)
    global_index && invpermute!(y, loc2glob(rtree))
    return y
end

function LinearAlgebra.ldiv!(L::HUnitLowerTriangular, y::AbstractVector)
    H = parent(L)
    if isleaf(H)
        d = data(H)
        ldiv!(UnitLowerTriangular(d), y) # B <-- L\B
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `ldiv`!"
        shift = pivot(H) .- 1
        chdH = children(H)
        m, n = size(chdH)
        @assert m === n
        for i in 1:m
            irows = colrange(chdH[i, i]) .- shift[2]
            bi = view(y, irows)
            for j in 1:(i-1)# j<i
                jrows = colrange(chdH[i, j]) .- shift[2]
                bj = view(y, jrows)
                _mul131!(bi, chdH[i, j], bj, -1)
            end
            # recursion stage
            ldiv!(UnitLowerTriangular(chdH[i, i]), bi)
        end
    end
    return y
end

function LinearAlgebra.ldiv!(U::HUpperTriangular, y::AbstractVector)
    H = parent(U)
    if isleaf(H)
        d = data(H)
        ldiv!(UpperTriangular(d), y) # B <-- L\B
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `ldiv`!"
        shift = pivot(H) .- 1
        chdH = children(H)
        m, n = size(chdH)
        @assert m === n
        for i in m:-1:1
            irows = colrange(chdH[i, i]) .- shift[2]
            bi = view(y, irows)
            for j in (i+1):n # j>i
                jrows = colrange(chdH[i, j]) .- shift[2]
                bj = view(y, jrows)
                _mul131!(bi, chdH[i, j], bj, -1)
            end
            # recursion stage
            ldiv!(UpperTriangular(chdH[i, i]), bi)
        end
    end
    return y
end
