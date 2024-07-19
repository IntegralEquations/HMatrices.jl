using DataFlowTasks

# helper function to wrap into the correct triangular type
_wraptriangular(d, ::UpperTriangular) = UpperTriangular(d)
_wraptriangular(d, ::LowerTriangular) = LowerTriangular(d)
_wraptriangular(d, ::UnitUpperTriangular) = UnitUpperTriangular(d)
_wraptriangular(d, ::UnitLowerTriangular) = UnitLowerTriangular(d)

isadmissible(H::HTriangular) = H |> parent |> isadmissible
data(t::HTriangular)         = _wraptriangular(data(parent(t)), t)
rowtree(H::HTriangular)      = H |> parent |> rowtree
coltree(H::HTriangular)      = H |> parent |> coltree
function children(T::HTriangular)
    H = parent(T)
    chdH = children(H)
    m, n = size(chdH)
    v = []
    for i in 1:m, j in 1:n
        chd = chdH[i, j]
        if i == j
            push!(v, _wraptriangular(chd, T))
        elseif i > j && T isa HLowerTriangular
            push!(v, chd)
        elseif j > i && T isa HUpperTriangular
            push!(v, chd)
        end
    end
    return v
end
parentnode(H::HTriangular)  = H |> parent |> parentnode |> adjoint
setdata!(H::HTriangular, d) = setdata!(parentnode(H), d)
isleaf(H::HTriangular)      = isempty(children(H))

hasdata(t::HTriangular) = hasdata(parent(t))

function Base.show(io::IO, ::MIME"text/plain", U::HUpperTriangular)
    H = parent(U)
    return println(io, "Upper triangular part of $H")
end

function Base.show(io::IO, ::MIME"text/plain", L::HLowerTriangular)
    H = parent(L)
    return println(io, "Lower triangular part of $H")
end

function LinearAlgebra.ldiv!(L::HLowerTriangular, B::AbstractMatrix)
    H = parent(L)
    if isleaf(H)
        d = data(H)
        ldiv!(_wraptriangular(d, L), B) # B <-- L\B
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `ldiv`!"
        shift = pivot(H) .- 1
        chdH = children(H)
        m, n = size(chdH)
        @assert m === n
        for i in 1:m
            irows = colrange(chdH[i, i]) .- shift[2]
            bi = view(B, irows, :)
            for j in 1:(i-1)# j<i
                jrows = colrange(chdH[i, j]) .- shift[2]
                bj = view(B, jrows, :)
                _mul131!(bi, chdH[i, j], bj, -1)
            end
            # recursion stage
            ldiv!(_wraptriangular(chdH[i, i], L), bi)
        end
    end
    return B
end

function LinearAlgebra.ldiv!(L::HLowerTriangular, R::RkMatrix)
    ldiv!(L, R.A) # change R.A in-place
    return R
end

function LinearAlgebra.ldiv!(
    L::HLowerTriangular,
    X::HMatrix,
    compressor,
    threads = false,
    bufs = nothing,
    level = 0,
    parentBlock = (0, 0, -1, -1),
)
    H = parent(L)
    @debug (isclean(H) || error("HMatrix is dirty"))
    if isleaf(X)
        if threads
            @dspawn begin
                @R(L)
                @RW(X)
                d = data(X)
                ldiv!(L, d)
            end label = "ldiv($(parentBlock[1]),$(parentBlock[2]))\nlvl=$(level)\np=($(parentBlock[3]),$(parentBlock[4]))"
        else
            d = data(X)
            ldiv!(L, d)
        end
    elseif isleaf(H) # X not a leaf, but L is a leaf. This should not happen.
        error()
    else
        chdH = children(H)
        chdX = children(X)
        m, n = size(chdH)
        @assert m == n
        for k in 1:size(chdX, 2)
            for i in 1:m
                for j in 1:(i-1)# j<i
                    hmul!(
                        chdX[i, k],
                        chdH[i, j],
                        chdX[j, k],
                        -1,
                        1,
                        compressor,
                        threads,
                        bufs,
                        level + 1,
                        (i, k, parentBlock[1], parentBlock[2]),
                    )
                end
                ldiv!(
                    _wraptriangular(chdH[i, i], L),
                    chdX[i, k],
                    compressor,
                    threads,
                    bufs,
                    level + 1,
                    (i, k, parentBlock[1], parentBlock[2]),
                )
            end
        end
    end
    return X
end

function LinearAlgebra.ldiv!(U::HUpperTriangular, B::AbstractMatrix)
    H = parent(U)
    if isleaf(H)
        d = data(H)
        ldiv!(_wraptriangular(d, U), B) # B <-- L\B
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `ldiv`!"
        shift = pivot(H) .- 1
        chdH = children(H)
        m, n = size(chdH)
        @assert m === n
        for i in m:-1:1
            irows = colrange(chdH[i, i]) .- shift[2]
            bi = view(B, irows, :)
            for j in (i+1):n # j>i
                jrows = colrange(chdH[i, j]) .- shift[2]
                bj = view(B, jrows, :)
                _mul131!(bi, chdH[i, j], bj, -1)
            end
            # recursion stage
            ldiv!(_wraptriangular(chdH[i, i], U), bi)
        end
    end
    return B
end

# 1.3
function LinearAlgebra.rdiv!(B::StridedMatrix, U::HUpperTriangular)
    H = parent(U)
    if isleaf(H)
        d = data(H)
        rdiv!(B, _wraptriangular(d, U)) # b <-- b/L
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `rdiv`!"
        shift = reverse(pivot(H) .- 1)
        chdH = children(H)
        m, n = size(chdH)
        @assert m === n
        for i in 1:m
            icols = rowrange(chdH[i, i]) .- shift[1]
            bi = view(B, :, icols)
            for j in 1:(i-1)
                jcols = rowrange(chdH[j, i]) .- shift[1]
                bj = view(B, :, jcols)
                _mul113!(bi, bj, chdH[j, i], -1)
            end
            # recursion stage
            rdiv!(bi, _wraptriangular(chdH[i, i], U))
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
function LinearAlgebra.rdiv!(
    X::HMatrix,
    U::HUpperTriangular,
    compressor,
    threads = false,
    bufs = nothing,
    level = 0,
    parentBlock = (0, 0, -1, -1),
)
    H = parent(U)
    if isleaf(X)
        if threads
            @dspawn begin
                @R(U)
                @RW(X)
                d = data(X)
                rdiv!(d, U) # b <-- b/L
            end label = "rdiv($(parentBlock[1]),$(parentBlock[2]))\nlvl=$(level)\np=($(parentBlock[3]),$(parentBlock[4]))"
        else
            d = data(X)
            rdiv!(d, U) # b <-- b/L
        end
    elseif isleaf(H)
        error()
    else
        @assert threads || !hasdata(H) # only leaves are allowed to have data for the inversion
        chdX = children(X)
        chdH = children(H)
        m, n = size(chdH)
        for k in 1:size(chdX, 1)
            for i in 1:m
                for j in 1:(i-1)
                    hmul!(
                        chdX[k, i],
                        chdX[k, j],
                        chdH[j, i],
                        -1,
                        1,
                        compressor,
                        threads,
                        bufs,
                        level + 1,
                        (k, i, parentBlock[1], parentBlock[2]),
                    )
                end
                rdiv!(
                    chdX[k, i],
                    _wraptriangular(chdH[i, i], U),
                    compressor,
                    threads,
                    bufs,
                    level + 1,
                    (k, i, parentBlock[1], parentBlock[2]),
                )
            end
        end
    end
    return X
end

function LinearAlgebra.ldiv!(L::HLowerTriangular, y::AbstractVector)
    H = parent(L)
    if isleaf(H)
        d = data(H)
        ldiv!(_wraptriangular(d, L), y) # B <-- L\B
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `ldiv`!"
        shift = pivot(H) .- 1
        chdH = children(H)
        m, n = size(chdH)
        @assert m === n
        for i in 1:m
            irows = colrange(chdH[i, i]) .- shift[2]
            bi = view(y, irows)
            for j in 1:(i-1)# j<i
                jrows = colrange(chdH[i, j]) .- shift[2]
                bj = view(y, jrows)
                _mul131!(bi, chdH[i, j], bj, -1)
            end
            # recursion stage
            ldiv!(_wraptriangular(chdH[i, i], L), bi)
        end
    end
    return y
end

function LinearAlgebra.ldiv!(U::HUpperTriangular, y::AbstractVector)
    H = parent(U)
    if isleaf(H)
        d = data(H)
        ldiv!(_wraptriangular(d, U), y) # B <-- L\B
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `ldiv`!"
        shift = pivot(H) .- 1
        chdH = children(H)
        m, n = size(chdH)
        @assert m === n
        for i in m:-1:1
            irows = colrange(chdH[i, i]) .- shift[2]
            bi = view(y, irows)
            for j in (i+1):n # j>i
                jrows = colrange(chdH[i, j]) .- shift[2]
                bj = view(y, jrows)
                _mul131!(bi, chdH[i, j], bj, -1)
            end
            # recursion stage
            ldiv!(_wraptriangular(chdH[i, i], U), bi)
        end
    end
    return y
end
