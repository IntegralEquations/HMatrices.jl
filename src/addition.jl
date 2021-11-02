"""
    (+)(A::Union{Matrix,RkMatrix,HMatrix},B::Union{Matrix,RkMatrix,HMatrix}) --> C

Two argument addition. When operating on `Union{Matrix,RkMatrix,HMatrix}`, the
result `C` is returned in the *natural format*, as described in the table below:

| `(+)(A,B)` | `B::Matrix`  | `B::RkMatrix`   | `B::HMatrix` |
|:-----|:---:|:---:|:---:|
|`A::Matrix`  | `C::Matrix`  | `C::Matrix` | `C::HMatrix` |
|`A::RkMatrix`  | `C::Matrix` | `C::RkMatrix` | `C::HMatrix` |
|`A::HMatrix`  | `C::HMatrix` | `C::HMatrix` | `C::HMatrix` |

You should use the `axpy!` method if you want the output in a format other
than the *natural* format.
"""

#1.2
Base.:(+)(M::Matrix,R::RkMatrix) = axpy!(1,M,Matrix(R))
Base.:(-)(M::Matrix,R::RkMatrix) = axpy!(-1,M,Matrix(R))

#1.3
Base.:(+)(M::Matrix,H::HMatrix)  = axpy!(true,M,deepcopy(H))
Base.:(-)(M::Matrix,H::HMatrix)  = axpy!(-1,M,deepcopy(H))

#2.1
Base.:(+)(R::RkMatrix,M::Matrix)  = (+)(M,R)
Base.:(-)(R::RkMatrix,M::Matrix)  = (-)(M,R)

#2.2
function Base.:(+)(R::RkMatrix,S::RkMatrix)
    Anew  = hcat(R.A,S.A)
    Bnew  = hcat(R.B,S.B)
    return RkMatrix(Anew,Bnew)
end
function Base.:(-)(R::RkMatrix,S::RkMatrix)
    Anew  = hcat(R.A,-S.A)
    Bnew  = hcat(R.B,S.B)
    return RkMatrix(Anew,Bnew)
end

#2.3
Base.:(+)(R::RkMatrix,H::HMatrix) = axpy!(1,R,deepcopy(H))
Base.:(-)(R::RkMatrix,H::HMatrix) = axpy!(-1,R,deepcopy(H))

#3.1
Base.:(+)(H::HMatrix,M::Matrix) = (+)(M,H)
Base.:(-)(H::HMatrix,M::Matrix) = (-)(M,H)

#3.2
Base.:(+)(H::HMatrix,R::RkMatrix) = (+)(R,H)
Base.:(-)(H::HMatrix,R::RkMatrix) = (-)(R,H)

#3.3
Base.:(+)(H::HMatrix,S::HMatrix) = axpy!(true,H,deepcopy(S))
Base.:(-)(H::HMatrix,S::HMatrix) = axpy!(-1,H,deepcopy(S))


"""
    LinearAlgebra.axpy!(a::Number,X::Union{Matrix,RkMatrix,HMatrix},Y::Union{Matrix,RkMatrix,HMatrix})

Perform `Y <-- a*X + Y` in-place. Note that depending on the types of `X` and
`Y`, this may require converting from/to different formats during intermdiate
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
function LinearAlgebra.axpy!(a,X::Matrix,Y::RkMatrix)
    axpy!(a,RkMatrix(X),Y)
end

# 1.3
function LinearAlgebra.axpy!(a,X::Matrix,Y::HMatrix)
    if hasdata(Y)
        axpy!(a,X,data(Y))
    else
        setdata!(Y,a*X)
    end
    return Y
end

# 2.1
function LinearAlgebra.axpy!(a,X::RkMatrix,Y::Matrix)
    axpy!(a,Matrix(X),Y)
end

#2.2
function LinearAlgebra.axpy!(a,X::RkMatrix,Y::RkMatrix)
    Y.A   = hcat(a*X.A,Y.A)
    Y.B   = hcat(X.B,Y.B)
    return Y
end

# 2.3
function LinearAlgebra.axpy!(a,X::RkMatrix,Y::HMatrix)
    if hasdata(Y)
        axpy!(a,X,data(Y))
    else
        setdata!(Y,a*X)
    end
    return Y
end

#3.1
function LinearAlgebra.axpy!(a,X::HMatrix,Y::Matrix)
    @debug "calling axpy! with `X` and HMatrix and `Y` a Matrix"
    shift = pivot(X) .- 1
    for block in PreOrderDFS(X)
        irange = rowrange(block) .- shift[1]
        jrange = colrange(block) .- shift[2]
        if hasdata(block)
            axpy!(a,data(block),view(Y,irange,jrange))
        end
    end
    return Y
end

# 3.2
function LinearAlgebra.axpy!(a,X::HMatrix,Y::RkMatrix)
    @debug "calling axpby! with `X` and HMatrix and `Y` an RkMatrix"
    # FIXME: inneficient implementation due to conversion from HMatrix to
    # Matrix. Does it really matter? I don't think this function should be
    # called.
    axpy!(a,Matrix(X),Y)
end

# 3.3
function LinearAlgebra.axpy!(a,X::HMatrix,Y::HMatrix)
    # TODO: assumes X and Y have the same structure. How to reinforce this?
    if hasdata(X)
        if hasdata(Y)
            axpy!(a,data(X),Y)
        else
            setdata!(Y,a*data(X))
        end
    end
    @assert size(children(X)) == size(children(Y)) "adding hierarchical matrices requires identical block structure"
    for (bx,by) in zip(children(X),children(Y))
        axpy!(a,bx,by)
    end
    return Y
end
