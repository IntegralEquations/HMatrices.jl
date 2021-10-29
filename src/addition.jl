# 1.1
# default axpby! on dense matrices

# 1.2
function LinearAlgebra.axpby!(a,X::Matrix,b,Y::RkMatrix)
    axpby!(a,RkMatrix(X),b,Y)
end

# 1.3
function LinearAlgebra.axpby!(a,X::Matrix,b,Y::HMatrix)
    @debug "should not be here"
    rmul!(Y,b)
    hasdata(Y) || error("target matrix must have a data field associated with it")
    axpby!(a,X,true,data(Y))
    return Y
end

# 2.1
function LinearAlgebra.axpby!(a,X::RkMatrix,b,Y::Matrix)
    axpby!(a,Matrix(X),b,Y)
end

#2.2
function LinearAlgebra.axpby!(a,X::RkMatrix,b,Y::RkMatrix)
    rmul!(Y,b)
    Y.A   = hcat(a*X.A,Y.A)
    Y.B   = hcat(X.B,Y.B)
    return Y
end

# 2.3
function LinearAlgebra.axpby!(a,X::RkMatrix,b,Y::HMatrix)
    rmul!(Y,b)
    hasdata(Y) || error("target matrix must have a data field associated with it")
    axpby!(a,X,true,data(Y))
    return Y
end

#3.1
function LinearAlgebra.axpby!(a,X::HMatrix,b,Y::Matrix)
    @debug "calling axpby! with `X` and HMatrix and `Y` a Matrix"
    rmul!(Y,b)
    shift = pivot(X) .- 1
    for block in PreOrderDFS(X)
        irange = rowrange(block) .- shift[1]
        jrange = colrange(block) .- shift[2]
        if hasdata(block)
            axpby!(a,data(block),true,view(Y,irange,jrange))
        end
    end
    return Y
end

# 3.2
function LinearAlgebra.axpby!(a,X::HMatrix,b,Y::RkMatrix)
    @debug "calling axpby! with `X` and HMatrix and `Y` an RkMatrix"
    R = RkMatrix(X)
    axpby!(a,R,b,Y)
end

# 3.3
function LinearAlgebra.axpby!(a,X::HMatrix,b,Y::HMatrix)
    # TODO: assumes X and Y have the same structure. How to reinforce this?
    rmul!(Y,b)
    if hasdata(X)
        if hasdata(Y)
            axpby!(a,data(X),true,Y)
        else
            setdata!(Y,a*data(X))
        end
    end
    for (bx,by) in zip(children(X),children(Y))
        axpby!(a,bx,true,by)
    end
    return Y
end
