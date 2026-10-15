const NOPIVOT = VERSION >= v"1.7" ? NoPivot : Val{false}

const HLU = LU{<:Any, <:HMatrix}

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
    _lu!(M, compressor, _buffer_channel(compressor, eltype(M)))
    # wrap the result in the LU structure
    return LU(M, Int[], 0)
end

"""
    lu!(M::HMatrix;atol=0,rank=typemax(Int),rtol=atol>0 ||
    rank<typemax(Int) ? 0 : sqrt(eps(Float64)))

Hierarhical LU facotrization of `M`, using the `PartialACA(;atol,rtol;rank)` compressor.
"""
function LinearAlgebra.lu!(M::HMatrix; threads = use_threads(), kwargs...)
    return lu!(M, PartialACA(; kwargs...); threads)
end

"""
    LinearAlgebra.lu(M::HMatrix,args...;kwargs...)

Hierarchical LU factorization. See [`lu!`](@ref) for the available options.
"""
LinearAlgebra.lu(M::HMatrix, args...; kwargs...) = lu!(deepcopy(M), args...; kwargs...)

function _lu!(M::HMatrix, compressor, bufs = nothing)
    if isleaf(M)
        d = data(M)
        @assert d isa Matrix
        lu!(d, NOPIVOT())
    else
        @assert !hasdata(M)
        chdM = children(M)
        m, n = size(chdM)
        for i in 1:m
            _lu!(chdM[i, i], compressor, bufs)
            for j in (i + 1):n
                ldiv!(UnitLowerTriangular(chdM[i, i]), chdM[i, j], compressor, bufs)
                rdiv!(chdM[j, i], UpperTriangular(chdM[i, i]), compressor, bufs)
            end
            for j in (i + 1):m
                for k in (i + 1):n
                    hmul!(chdM[j, k], chdM[j, i], chdM[i, k], -1, 1, compressor, bufs)
                end
            end
        end
    end
    return M
end
