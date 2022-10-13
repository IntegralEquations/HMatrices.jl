const NOPIVOT = VERSION >= v"1.7" ? NoPivot : Val{false}

function Base.getproperty(LU::LU{<:Any,<:HMatrix}, s::Symbol)
    H = getfield(LU, :factors) # the underlying hierarchical matrix
    if s == :L
        return UnitLowerTriangular(H)
    elseif s == :U
        return UpperTriangular(H)
    else
        return getfield(LU, s)
    end
end

"""
    lu!(M::HMatrix[;atol,rank,rtol])
    lu!(M::HMatrix,comp::AbstractCompressor)

In-place hierarhical LU facotrization of `M`; the keyword arguemnts `atol`,
`rank`, and `rtol` can be used to control the quality of the approximation.
Alternatively, you  may directly pass an `AbstractCompressor` to compress
low-rank blocks during the multiplication steps.

By default, the factorization is multithreaded through the use of
`DataFlowTasks`; to disable it, you must call
`DataFlowTasks.force_sequential(true)`.
"""
function lu!(M::HMatrix;
             atol=0,
             rank=typemax(Int),
             rtol=atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
             kwargs...)
    compressor = PartialACA(atol, rank, rtol)
    return lu!(M, compressor; kwargs...)
end
function lu!(M::HMatrix, compressor)
    # perform the lu decomposition of M in place
    @timeit_debug "lu factorization" begin
        _lu!(M, compressor)
    end
    # wrap the result in the LU structure
    res = @dspawn LU(@R(M), LinearAlgebra.BlasInt[], LinearAlgebra.BlasInt(0)) label = "LU"
    return fetch(res)
end

"""
    lu(M::HMatrix,args...;kwargs...)

Hierarchical LU factorization. See [`lu!`](@ref) for the available options.
"""
lu(M::HMatrix, args...; kwargs...) = lu!(deepcopy(M), args...; kwargs...)

function _lu!(M::HMatrix, compressor)
    if isleaf(M)
        @dspawn lu!(data(@RW(M)), NOPIVOT()) label = "Dense LU"
    else
        chdM = children(M)
        m, n = size(chdM)
        for i in 1:m
            _lu!(chdM[i, i], compressor)
            for j in (i + 1):n
                ldiv!(UnitLowerTriangular(chdM[i, i]), chdM[i, j],
                      compressor)
                rdiv!(chdM[j, i], UpperTriangular(chdM[i, i]),
                      compressor)
            end
            for j in (i + 1):m
                for k in (i + 1):n
                    hmul!(chdM[j, k], chdM[j, i], chdM[i, k], -1, 1, compressor, Val(false))
                end
            end
        end
    end
    return M
end

# routines needed to solve linear problem as HLU\y
function ldiv!(A::LU{<:Any,<:HMatrix}, y::AbstractVector; global_index=true)
    H = A.factors # underlying data
    ctree = coltree(H)
    rtree = rowtree(H)
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

function ldiv!(L::HUnitLowerTriangular, y::AbstractVector)
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
            for j in 1:(i - 1)# j<i
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

function ldiv!(U::HUpperTriangular, y::AbstractVector)
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
            for j in (i + 1):n # j>i
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
