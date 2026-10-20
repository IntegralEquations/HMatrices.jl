const HLU = LU{<:Any, <:HMatrix}

function Base.getproperty(LU::HLU, s::Symbol)
    H = getfield(LU, :factors) # the underlying hierarchical matrix
    if s == :L
        return UnitLowerTriangular(H)
    elseif s == :U
        return UpperTriangular(H)
    elseif s == :p
        return rowperm(H)
    elseif s == :q
        return colperm(H)
    else
        return getfield(LU, s)
    end
end

Base.propertynames(::HLU, private::Bool = false) = (:L, :U, :p, :q, (private ? fieldnames(LU) : ())...)

function Base.show(io::IO, ::MIME"text/plain", LU::HLU)
    H = getfield(LU, :factors) # the underlying hierarchical matrix
    return println(io, "LU factorization of $H")
end

"""
    lu!(M::HMatrix,comp)

Hierarchical LU factorization of `M`, using `comp` to generate the compressed
blocks during the multiplication routines.

`F \\ b` solves in the ordering of the points used to assemble `M`. The factors `F.L`
and `F.U` are stored in the ordering of the cluster trees, with
`F.L * (F.U * x) ≈ Matrix(M)[F.p, F.q] * x`; products and solves with them act in that ordering.

The factorization is currently serial: `threads` is accepted but not used.
"""
function LinearAlgebra.lu!(M::HMatrix, compressor; threads = true)
    # perform the lu decomposition of M in place
    _lu!(M, compressor, _buffer_channel(compressor, eltype(M)))
    # wrap the result in the LU structure
    return LU(M, Int[], 0)
end

"""
    lu!(M::HMatrix; atol, rank, rtol, threads = true)

Hierarchical LU factorization of `M`, using the `PartialACA(; atol, rank, rtol)` compressor.
"""
function LinearAlgebra.lu!(M::HMatrix; threads = true, kwargs...)
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
        lu!(d, NoPivot())
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
                    _hmul_tree!(chdM[j, k], chdM[j, i], chdM[i, k], -1, 1, compressor, bufs)
                end
            end
        end
    end
    return M
end
