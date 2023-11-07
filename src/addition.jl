"""
    axpy!(a::Number,X::Union{Matrix,RkMatrix,HMatrix},Y::Union{Matrix,RkMatrix,HMatrix})

Perform `Y <-- a*X + Y` in-place. Note that depending on the types of `X` and
`Y`, this may require converting from/to different formats during intermediate
calculations.

In the case where `Y` is an `RkMatrix`, the call `axpy!(a,X,Y)` should
typically be followed by recompression stage to keep the rank of `Y` under
control.

In the case where `Y` is an `HMatrix`, the call `axpy!(a,X,Y)` sums `X` to the
data in the node `Y` (and not on the leaves). In case `Y` has no `data`, it will
simply be assigned `X`. This means that after the call `axpy(a,X,Y)`, the object
`Y` is in a *dirty* state (see [`isclean`][@ref]) and usually a call to
[`flush_to_leaves!`](@ref) or [`flush_to_children!`](@ref) follows.
"""
function LinearAlgebra.axpy!(a, X::Base.Matrix, Y::RkMatrix)
    return axpy!(a, RkMatrix(X), Y)
end

# 1.3
function LinearAlgebra.axpy!(a, X::Base.Matrix, Y::HMatrix)
    if hasdata(Y)
        axpy!(a, X, data(Y))
    else
        setdata!(Y, a * X)
    end
    return Y
end

# 2.1
function LinearAlgebra.axpy!(a, X::RkMatrix, Y::Base.Matrix)
    # axpy!(a,Matrix(X),Y)
    r = rank(X)
    m, n = size(Y)
    # @turbo warn_check_args=false for i in 1:m
    @inbounds for i in 1:m, j in 1:n, k in 1:r
        Y[i,j] += a * X.A[i, k] * adjoint(X.B[j, k])
    end
    return Y
end

#2.2
function LinearAlgebra.axpy!(a, X::RkMatrix, Y::RkMatrix)
    Y.A = hcat(a * X.A, Y.A)
    Y.B = hcat(X.B, Y.B)
    return Y
end

# 2.3
function LinearAlgebra.axpy!(a, X::RkMatrix, Y::HMatrix)
    if hasdata(Y)
        axpy!(a, X, data(Y))
    else
        setdata!(Y, a * X)
    end
    return Y
end

#3.1
function LinearAlgebra.axpy!(a, X::HMatrix, Y::Matrix)
    @debug "calling axpy! with `X` and HMatrix and `Y` a Matrix"
    shift = pivot(X) .- 1
    for block in AbstractTrees.PreOrderDFS(X)
        irange = rowrange(block) .- shift[1]
        jrange = colrange(block) .- shift[2]
        if hasdata(block)
            axpy!(a, data(block), view(Y, irange, jrange))
        end
    end
    return Y
end

# 3.2
function LinearAlgebra.axpy!(a, X::HMatrix, Y::RkMatrix)
    @debug "calling axpby! with `X` and HMatrix and `Y` an RkMatrix"
    # FIXME: inneficient implementation due to conversion from HMatrix to
    # Matrix. Does it really matter? I don't think this function should be
    # called.
    return axpy!(a, Matrix(X; global_index = false), Y)
end

# 3.3
function LinearAlgebra.axpy!(a, X::HMatrix, Y::HMatrix)
    # TODO: assumes X and Y have the same structure. How to reinforce this?
    if hasdata(X)
        if hasdata(Y)
            axpy!(a, data(X), Y)
        else
            setdata!(Y, a * data(X))
        end
    end
    @assert size(children(X)) == size(children(Y)) "adding hierarchical matrices requires identical block structure"
    for (bx, by) in zip(children(X), children(Y))
        axpy!(a, bx, by)
    end
    return Y
end

# add a unifor scaling to an HMatrix return an HMatrix
function LinearAlgebra.axpy!(a, X::UniformScaling, Y::HMatrix)
    @assert isclean(Y)
    if hasdata(Y)
        d = data(Y)
        @assert d isa Matrix
        n = min(size(d)...)
        for i in 1:n
            d[i, i] += a * X.λ
        end
    else
        n = min(blocksize(Y)...)
        for i in 1:n
            axpy!(a, X, children(Y)[i, i])
        end
    end
    return Y
end

Base.:(+)(X::UniformScaling, Y::HMatrix) = axpy!(true, X, deepcopy(Y))
Base.:(+)(X::HMatrix, Y::UniformScaling) = Y + X
