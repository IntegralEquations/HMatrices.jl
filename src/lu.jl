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

function LinearAlgebra.lu!(M::HMatrix,posthook)
    #perform the lu decomposition of M in place
    @timeit_debug "lu factorization" begin
        _lu!(M,posthook)
    end
    #wrap the result in the LU structure
    return LU(M)
end

LinearAlgebra.lu(M::HMatrix,posthook) = lu!(deepcopy(M),posthook)

function _lu!(M::HMatrix,posthook)
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
            _lu!(chdM[i,i],posthook)
            for j=i+1:n
                @timeit_debug "ldiv! solution" begin
                    ldiv!(UnitLowerTriangular(chdM[i,i]),chdM[i,j],posthook)
                end
                @timeit_debug "rdiv! solution" begin
                    rdiv!(chdM[j,i],UpperTriangular(chdM[i,i]),posthook)
                end
                @timeit_debug "hmul!" begin
                    hmul!(chdM[j,j],chdM[j,i],chdM[i,j],-1,1,posthook)
                end
            end
        end
    end
    return M
end
