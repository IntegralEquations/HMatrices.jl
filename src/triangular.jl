# helper function to wrap into the correct triangular type
_wraptriangular(d, ::UpperTriangular) = UpperTriangular(d)
_wraptriangular(d, ::LowerTriangular) = LowerTriangular(d)
_wraptriangular(d, ::UnitUpperTriangular) = UnitUpperTriangular(d)
_wraptriangular(d, ::UnitLowerTriangular) = UnitLowerTriangular(d)
# A triangle of a `Hermitian` leaf is a triangle of the stored data, or the adjoint of one.
# Unwrapping it lets `ldiv!` use BLAS (trsm/trsv) instead of the generic fallback that goes
# through `getindex` of the `Hermitian` wrapper (e.g. for `cholesky(H).L`, which is the lower
# triangle of a `Hermitian` whose data is the upper factor).
function _wraptriangular(d::Hermitian, ::LowerTriangular)
    return d.uplo == 'U' ? UpperTriangular(parent(d))' : LowerTriangular(parent(d))
end
function _wraptriangular(d::Hermitian, ::UpperTriangular)
    return d.uplo == 'U' ? UpperTriangular(parent(d)) : LowerTriangular(parent(d))'
end

isadmissible(H::HTriangular) = H |> parent |> isadmissible
data(t::HTriangular) = _wraptriangular(data(parent(t)), t)
rowtree(H::HTriangular) = H |> parent |> rowtree
coltree(H::HTriangular) = H |> parent |> coltree
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
# the diagonal children of an internal node are always in the triangle
isleaf(H::HTriangular) = isleaf(parent(H))

hasdata(t::HTriangular) = hasdata(parent(t))

function Base.show(io::IO, tri::HTriangular)
    part = tri isa HUpperTriangular ? "Upper" : "Lower"
    return println(io, "$part triangular part of $(parent(tri))")
end
Base.show(io::IO, ::MIME"text/plain", tri::HTriangular) = show(io, tri)

function LinearAlgebra.ldiv!(L::HLowerTriangular, B::AbstractVecOrMat)
    return _ldiv!(L, B, promote_type(eltype(L), eltype(B))[])
end

"""
    _ldiv!(L::HLowerTriangular, B, buf)

`B ← L⁻¹B` by block forward substitution, with the products with the low-rank blocks
of `L` in the scratch vector `buf` (see `_leaf_gemm!`).
"""
function _ldiv!(L::HLowerTriangular, B, buf)
    H = parent(L)
    if isleaf(H)
        d = data(H)
        ldiv!(_wraptriangular(d, L), B) # B <-- L\B
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `ldiv`!"
        shift = offset(H)
        chdH = children(H)
        m, n = size(chdH)
        @assert m === n
        for i in 1:m
            irows = colrange(chdH[i, i]) .- shift[2]
            bi = selectdim(B, 1, irows)
            for j in 1:(i - 1) # j<i
                jrows = colrange(chdH[i, j]) .- shift[2]
                bj = selectdim(B, 1, jrows)
                _mul131!(bi, chdH[i, j], bj, -1, buf)
            end
            # recursion stage
            _ldiv!(_wraptriangular(chdH[i, i], L), bi, buf)
        end
    end
    return B
end

function LinearAlgebra.ldiv!(L::HLowerTriangular, R::RkMatrix)
    ldiv!(L, R.A) # change R.A in-place
    return _modified_factor!(R, :A)
end

function LinearAlgebra.ldiv!(U::HUpperTriangular, B::AbstractVecOrMat)
    return _ldiv!(U, B, promote_type(eltype(U), eltype(B))[])
end

"""
    _ldiv!(U::HUpperTriangular, B, buf)

`B ← U⁻¹B` by block backward substitution; see `_ldiv!` for a lower triangle.
"""
function _ldiv!(U::HUpperTriangular, B, buf)
    H = parent(U)
    if isleaf(H)
        d = data(H)
        ldiv!(_wraptriangular(d, U), B) # B <-- U\B
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `ldiv`!"
        shift = offset(H)
        chdH = children(H)
        m, n = size(chdH)
        @assert m === n
        for i in m:-1:1
            irows = colrange(chdH[i, i]) .- shift[2]
            bi = selectdim(B, 1, irows)
            for j in (i + 1):n # j>i
                jrows = colrange(chdH[i, j]) .- shift[2]
                bj = selectdim(B, 1, jrows)
                _mul131!(bi, chdH[i, j], bj, -1, buf)
            end
            # recursion stage
            _ldiv!(_wraptriangular(chdH[i, i], U), bi, buf)
        end
    end
    return B
end

function LinearAlgebra.rdiv!(B::StridedMatrix, U::HUpperTriangular)
    H = parent(U)
    if isleaf(H)
        d = data(H)
        rdiv!(B, _wraptriangular(d, U)) # B <-- B/U
    else
        @assert !hasdata(H) "only leaves are allowed to have data when using `rdiv`!"
        shift = reverse(offset(H))
        chdH = children(H)
        m, n = size(chdH)
        @assert m === n
        for i in 1:m
            icols = rowrange(chdH[i, i]) .- shift[1]
            bi = view(B, :, icols)
            for j in 1:(i - 1)
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

"""
    rdiv!(R::RkMatrix, U::HUpperTriangular[, buf::Vector])

`R ← R*U⁻¹`, forming the transposed factor in the scratch vector `buf` (grown if
needed).
"""
function LinearAlgebra.rdiv!(R::RkMatrix, U::HUpperTriangular, buf::Vector = eltype(R)[])
    Bt = adjoint!(_scratch!(buf, reverse(size(R.B))), R.B)
    adjoint!(R.B, rdiv!(Bt, U))
    return _modified_factor!(R, :B)
end

"""
    hldiv!(L::HLowerTriangular, X::HMatrix, compressor; executor = SerialExecutor())

`X ← L⁻¹X` for a triangle `L` of an `HMatrix` (or of its adjoint), compressing
every admissible leaf of `X` once with `compressor` (see [`LazyDriver`](@ref)), the
numeric work being run by `executor`.
"""
function hldiv!(L::HLowerTriangular, X::HMatrix, compressor; executor = SerialExecutor())
    d = LazyDriver(X, _checkblocks(parent(L)), X, compressor; executor)
    _run!(() -> _solve_lower!(d, L, X), executor)
    return X
end

"""
    hrdiv!(X::HMatrix, U::HUpperTriangular, compressor; executor = SerialExecutor())

`X ← XU⁻¹` for a triangle `U` of an `HMatrix` (or of its adjoint); see [`hldiv!`](@ref).
"""
function hrdiv!(X::HMatrix, U::HUpperTriangular, compressor; executor = SerialExecutor())
    d = LazyDriver(X, X, _checkblocks(parent(U)), compressor; executor)
    _run!(() -> _solve_upper!(d, X, U), executor)
    return X
end

_checkblocks(H::Union{HMatrix, AdjointHMatrix}) = H
_checkblocks(H) = throw(ArgumentError("triangular solves with an `HMatrix` need a triangle of an `HMatrix` or of its adjoint"))
