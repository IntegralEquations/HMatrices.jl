function inv!(M::HMatrix,X,compressor)
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
        inv!(chdM[1,1],chdX[1,1],compressor)
        #update
        hmul!(chdX[1,2],chdM[1,1],chdM[1,2],-1,false,compressor)
        hmul!(chdX[2,1],chdM[2,1],chdM[1,1],true,false,compressor)
        hmul!(chdM[2,2],chdM[2,1],chdX[1,2],true,true,compressor)
        # recursion
        inv!(chdM[2,2],chdX[2,2],compressor)
        # update
        hmul!(chdM[1,2],chdX[1,2],chdM[2,2],1,0,compressor)
        hmul!(chdM[1,1],chdM[1,2],chdX[2,1],-1,true,compressor)
        hmul!(chdM[2,1],chdM[2,2],chdX[2,1],-1,false,compressor)
    end
    return M
end

function LinearAlgebra.inv(M::HMatrix)
    compressor = TSVD(;atol=1e-10)
    inv!(deepcopy(M),zero(M),compressor)
end

function LinearAlgebra.inv(M::HMatrix,posthook)
    inv!(deepcopy(M),zero(M),posthook)
end
