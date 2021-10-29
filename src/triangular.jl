const HUnitLowerTriangular = UnitLowerTriangular{<:Any,<:HMatrix}
const HUpperTriangular     = UpperTriangular{<:Any,<:HMatrix}

# 3.1
function LinearAlgebra.ldiv!(L::HUnitLowerTriangular, B::AbstractMatrix)
    H = parent(L)
    if isleaf(H)
        d = data(H)
        ldiv!(UnitLowerTriangular(d), B) # B <-- L\B
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `ldiv`!"
        shift = pivot(H) .- 1
        chdH  = children(H)
        m, n   = size(chdH)
        @assert m === n
        for i = 1:m
            irows  = colrange(chdH[i,i]) .- shift[2]
            bi     = view(B, irows, :)
            for j = 1:(i - 1)# j<i
                jrows  = colrange(chdH[i,j]) .- shift[2]
                bj     = view(B, jrows, :)
                mul!(bi, chdH[i,j], bj, -1, true)
            end
            # recursion stage
            ldiv!(UnitLowerTriangular(chdH[i,i]), bi)
        end
    end
    return B
end

# 3.2
function LinearAlgebra.ldiv!(L::HUnitLowerTriangular, R::RkMatrix)
    ldiv!(L, R.A) # change R.A in-place
    return R
end

function LinearAlgebra.ldiv!(L::HUnitLowerTriangular, X::HMatrix, posthook)
    H = parent(L)
    @assert isclean(H)
    if isleaf(X)
        d = data(X)
        ldiv!(L, d)
    elseif isleaf(H) # X not a leaf, but L is a leaf. This should not happen.
        error()
    else
        chdH = children(H)
        chdX = children(X)
        m, n  = size(chdH)
        @assert m == n
        for k in 1:size(chdX, 2)
            for i = 1:m
                for j = 1:(i - 1)# j<i
                    hmul!(chdX[i,k], chdH[i,j], chdX[j,k], -1, true, posthook)
                end
                ldiv!(UnitLowerTriangular(chdH[i,i]), chdX[i,k], posthook)
            end
        end
    end
    return X
end

# 1.3
function LinearAlgebra.rdiv!(B::StridedMatrix, U::HUpperTriangular)
    H = parent(U)
    if isleaf(H)
        d = data(H)
        rdiv!(B, UpperTriangular(d)) # b <-- b/L
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `rdiv`!"
        shift = pivot(H) .- 1 |> reverse
        chdH  = children(H)
        m, n   = size(chdH)
        @assert m === n
        for i = 1:m
            icols  = rowrange(chdH[i,i]) .- shift[1]
            bi     = view(B, :, icols)
            for j = 1:(i - 1)
                jcols  = rowrange(chdH[j,i]) .- shift[1]
                bj     = view(B, :, jcols)
                mul!(bi, bj, chdH[j,i], -1, true)
            end
            # recursion stage
            rdiv!(bi, UpperTriangular(chdH[i,i]))
        end
    end
    return B
end

# 2.3
function LinearAlgebra.rdiv!(R::RkMatrix, U::HUpperTriangular)
    Bt = rdiv!(Matrix(R.Bt), U)
    adjoint!(R.B, Bt)
    return R
end

# 3.3
function LinearAlgebra.rdiv!(X::AbstractHMatrix, U::HUpperTriangular, posthook)
    H = parent(U)
    if isleaf(X)
        d = data(X)
        rdiv!(d, U) # b <-- b/L
    elseif isleaf(H)
        error()
    else
        @assert !hasdata(H) # only leaves are allowed to have data for the inversion
        chdX = children(X)
        chdH = children(H)
        m, n = size(chdH)
        for k in 1:size(chdX, 1)
            for i = 1:m
                for j = 1:(i - 1)
                    hmul!(chdX[k,i], chdX[k,j], chdH[j,i], -1, true, posthook)
                end
                rdiv!(chdX[k,i], UpperTriangular(chdH[i,i]), posthook)
            end
        end
    end
    return X
end
