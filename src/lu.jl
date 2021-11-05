LinearAlgebra.LU(H::HMatrix) = LU(H,Int[],0)

const HLU = LU{<:Any,<:HMatrix}

function Base.getproperty(LU::HLU,s::Symbol)
    H = getfield(LU,:factors) # the underlying hierarchical matrix
    if s == :L
        return UnitLowerTriangular(H)
    elseif s == :U
        return UpperTriangular(H)
    else
        return getfield(LU,s)
    end
end

function LinearAlgebra.lu!(M::HMatrix,compressor)
    #perform the lu decomposition of M in place
    @timeit_debug "lu factorization" begin
        _lu!(M,compressor)
    end
    #wrap the result in the LU structure
    return LU(M)
end

function LinearAlgebra.lu!(M::HMatrix;atol=0,rank=typemax(Int),rtol=atol>0 || rank<typemax(Int) ? 0 : sqrt(eps(Float64)))
    compressor = PartialACA(atol,rank,rtol)
    lu!(M,compressor)
end

LinearAlgebra.lu(M::HMatrix,args...;kwargs...) = lu!(deepcopy(M),args...;kwargs...)

function _lu!(M::HMatrix,compressor)
    if isleaf(M)
        d = data(M)
        @assert d isa Matrix
        @timeit_debug "dense lu factorization" begin
            lu!(d,Val(false)) # Val(false) for pivot of dense lu factorization
        end
    else
        @assert !hasdata(M)
        chdM = children(M)
        m,n = size(chdM)
        for i=1:m
            _lu!(chdM[i,i],compressor)
            for j=i+1:n
                @timeit_debug "ldiv! solution" begin
                    ldiv!(UnitLowerTriangular(chdM[i,i]),chdM[i,j],compressor)
                end
                @timeit_debug "rdiv! solution" begin
                    rdiv!(chdM[j,i],UpperTriangular(chdM[i,i]),compressor)
                end
                @timeit_debug "hmul!" begin
                    hmul!(chdM[j,j],chdM[j,i],chdM[i,j],-1,1,compressor)
                end
            end
        end
    end
    return M
end

function LinearAlgebra.ldiv!(A::HLU,y::AbstractVector;global_index=true)
    p         = A.factors # underlying data
    ctree     = coltree(p)
    rtree     = rowtree(p)
    # permute input
    global_index && permute!(y,loc2glob(ctree))
    L,U = A.L, A.U
    # solve LUx = y through:
    # (a) L(z) = y
    # (b) Ux   = z
    ldiv!(L,y)
    ldiv!(U,y)
    global_index && invpermute!(y,loc2glob(rtree))
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
        chdH  = children(H)
        m, n   = size(chdH)
        @assert m === n
        for i = 1:m
            irows  = colrange(chdH[i,i]) .- shift[2]
            bi     = view(y, irows)
            for j = 1:(i - 1)# j<i
                jrows  = colrange(chdH[i,j]) .- shift[2]
                bj     = view(y, jrows)
                _mul131!(bi, chdH[i,j], bj, -1)
            end
            # recursion stage
            ldiv!(UnitLowerTriangular(chdH[i,i]), bi)
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
        chdH  = children(H)
        m, n   = size(chdH)
        @assert m === n
        for i = m:-1:1
            irows  = colrange(chdH[i,i]) .- shift[2]
            bi     = view(y, irows)
            for j = i+1:n # j>i
                jrows  = colrange(chdH[i,j]) .- shift[2]
                bj     = view(y, jrows)
                _mul131!(bi, chdH[i,j], bj, -1)
            end
            # recursion stage
            ldiv!(UpperTriangular(chdH[i,i]), bi)
        end
    end
    return y
end
