function inv!(M::HMatrix,X,posthook)
    @assert isclean(M) "input matrix can have data only at leaves for inversion"
    T = eltype(M)
    chdM = children(M)
    chdX = children(X)
    if isleaf(M)
        # only the diagonal blocks get here
        d = data(M)
        @assert d isa Matrix
        setdata!(M,inv(d))
    else
        @assert size(chdM) == size(chdX) == (2,2)
        #recursion
        inv!(chdM[1,1],chdX[1,1],posthook)
        #update
        hmul!(chdX[1,2],chdM[1,1],chdM[1,2],-1,false,posthook)
        hmul!(chdX[2,1],chdM[2,1],chdM[1,1],true,false,posthook)
        hmul!(chdM[2,2],chdM[2,1],chdX[1,2],true,true,posthook)
        # recursion
        inv!(chdM[2,2],chdX[2,2],posthook)
        # update
        hmul!(chdM[1,2],chdX[1,2],chdM[2,2],1,0,posthook)
        hmul!(chdM[1,1],chdM[1,2],chdX[2,1],-1,true,posthook)
        hmul!(chdM[2,1],chdM[2,2],chdX[2,1],-1,false,posthook)
    end
    return M
end

function LinearAlgebra.inv(M::HMatrix)
    posthook = MulPostHook(TSVD(;atol=1e-10))
    # posthook = MulPostHook(PartialACA(;atol=1e-10))
    # posthook = (block) -> begin
    #     if block isa HMatrix
    #         flush_to_children!(block)
    #     else
    #         block
    #     end
    # end
    inv!(deepcopy(M),zero(M),posthook)
end

function LinearAlgebra.inv(M::HMatrix,posthook)
    inv!(deepcopy(M),zero(M),posthook)
end
