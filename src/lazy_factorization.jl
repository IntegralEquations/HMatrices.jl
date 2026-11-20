"""
    mutable struct LazyLowRankTerm{T}

An exact low-rank term `A*B'` of the size of the block it was computed for: the
product `α*X*Y` of two blocks of the factors, one of them being a leaf. One of `A`
and `B` is usually a factor of this leaf (not a copy), the other one is computed.

It is computed by a piece of work (see `_compute_low_rank_term`), and read by the
flushes of the leaves below the block (see [`LazyDriver`](@ref)), through
restrictions ([`CarryTerm`](@ref)).
"""
mutable struct LazyLowRankTerm{T}
    A::Matrix{T}
    B::Matrix{T}
end
LazyLowRankTerm{T}() where {T} = LazyLowRankTerm{T}(Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0))

"""
    struct CarryTerm{T}

Restriction of a [`LazyLowRankTerm`](@ref) to a block below the one it was computed
for: its rows starting after `row_offset`, and its columns starting after
`col_offset`, the size of the restriction being the one of this block.
"""
struct CarryTerm{T}
    term::LazyLowRankTerm{T}
    row_offset::Int
    col_offset::Int
end

"""
    factors(t::CarryTerm, m, n)

The factors `(A, B)` of the restriction `t = A*B'`, of size `m × n`, as views.
"""
factors(t::CarryTerm, m, n) = (view(t.term.A, t.row_offset .+ (1:m), :), view(t.term.B, t.col_offset .+ (1:n), :))

"""
    struct PendingUpdates{P,T}

The updates pending at a node: the products `(X, Y)` (of type `P`) it still owes, and
the exact low-rank terms already evaluated for it (the carry).
"""
struct PendingUpdates{P, T}
    products::Vector{P}
    carry::Vector{CarryTerm{T}}
end
PendingUpdates{P, T}() where {P, T} = PendingUpdates{P, T}(P[], CarryTerm{T}[])

"""
    struct UpdateBuffer{B,T}

Workspace of a flush: the buffer `comp` of the compressor, the storage `lowrank` of
the packed low-rank factors of the leaf being compressed (see
`_flush_low_rank_leaf!`), and two vectors `scratch` and `rkscratch` for intermediate
products (those of a [`LeafOperator`](@ref)). The vectors grow as needed and are
reused.
"""
struct UpdateBuffer{B, T}
    comp::B
    lowrank::Vector{T}
    scratch::Vector{T}
    rkscratch::Vector{T}
end
UpdateBuffer(comp, T::Type) = UpdateBuffer(allocate_buffer(comp, T), T[], T[], T[])

"""
    struct LazyDriver

State of an algorithm on an `HMatrix` where every update is deferred until the block
it targets is factorized, solved for, or flushed (e.g. the [`lu!`](@ref) and
[`cholesky!`](@ref) factorizations, `hmul!`), with its numeric work run by an
executor ([`SerialExecutor`](@ref) or [`ThreadsExecutor`](@ref)).

By the block Crout identity, every block of the factors is determined by a single
Schur complement, a sum of products of blocks of the factors that precede it.
Instead of evaluating these products one at a time and recompressing after each
one, each node carries the products it still owes, and the exact low-rank terms
already evaluated for it (see [`PendingUpdates`](@ref)), representing the update
`C ← C + Σₖ AₖBₖ' + α*Σᵢ XᵢYᵢ`. A node is flushed right before it is factorized or
solved for (see [`flush!`](@ref)), so that every admissible leaf is compressed once,
with `comp`. This is the accumulator of Börm (2019, "Hierarchical matrix arithmetic
with accumulated updates"), except that an admissible leaf is compressed once from
the whole sum.

The driver does the structural work (pushing and splitting products, restricting
low-rank terms) itself, and hands the numeric work to `executor`: the low-rank
terms of a node, the flush of a leaf, and the factorization of a diagonal leaf or
the triangular solve on a leaf. A leaf is written by at most one flush followed by
one factorization or solve, a low-rank term by one piece of work, and a block is
only read once final, which is the access pattern of the executors. The workspaces
of the pieces of work are taken from `buffers`.

With `uplo = 'U'` (or `'L'`), only the upper (or lower) triangle of the diagonal
blocks (whose rows and columns have the same indices) is updated, as for the data of a
`Hermitian`; `uplo = 'N'` updates every block.

    LazyDriver(C::HMatrix, X, Y, comp; α = -1, uplo = 'N', executor = SerialExecutor())

A driver for the nodes of `C` (and of any `HMatrix` of the same type), with products
of blocks of the types of `X` and `Y` (`HMatrix` or adjoint).
"""
struct LazyDriver{S, T, P, A, C, E, B}
    pending::IdDict{HMatrix{S, T}, PendingUpdates{P, T}}
    α::A
    uplo::Char
    comp::C
    executor::E
    buffers::B
end
function LazyDriver(::HMatrix{S, T}, X, Y, comp; α = -1, uplo = 'N', executor = SerialExecutor()) where {S, T}
    # once here rather than in each block, where it would be raised inside a task
    T <: Number || throw(ArgumentError("factorizations, products and solves of `HMatrix`s need scalar entries, not $T"))
    _check_rtol(comp, T)
    P = Tuple{typeof(X), typeof(Y)}
    buffers = workspaces(UpdateBuffer{typeof(allocate_buffer(comp, T)), T})
    return LazyDriver(IdDict{HMatrix{S, T}, PendingUpdates{P, T}}(), α, uplo, comp, executor, buffers)
end

"""
    pending!(d::LazyDriver, C::HMatrix)

The [`PendingUpdates`](@ref) of `C`, created if `C` has none.
"""
function pending!(d::LazyDriver{S, T, P}, C::HMatrix) where {S, T, P}
    return get!(PendingUpdates{P, T}, d.pending, C)
end

"""
    push!(d::LazyDriver, C::HMatrix, X, Y)

Register the update `C ← C + α*X*Y` (see [`LazyDriver`](@ref)), without performing
any arithmetic. `X` and `Y` must not be modified afterwards: their data may be
referenced (without copies) by the low-rank terms created when the update is
applied.
"""
function Base.push!(d::LazyDriver, C::HMatrix, X, Y)
    push!(pending!(d, C).products, (X, Y))
    return d
end

"""
    with_buffer(f, d::LazyDriver)

`f(buf)` with a workspace `buf` (an [`UpdateBuffer`](@ref)) used by no other piece
of work at the same time (see [`with_workspace`](@ref)).
"""
with_buffer(f, d::LazyDriver{S, T}) where {S, T} = with_workspace(f, d.buffers, () -> UpdateBuffer(d.comp, T))

"""
    flush!(d::LazyDriver, C::HMatrix)

Apply the updates pending at `C` (not those pending at its descendants):

- internal node: a pending product with a leaf factor has an exact low-rank form,
  which is computed once, here, and shared by the leaves below (carry); the other
  products are split into the products of the children; the carry is restricted to
  the children;
- dense leaf: the carry and the products are added exactly;
- admissible leaf: its data, its carry and the products with an exact low-rank form
  are written side by side as `A*B'`, and `A*B' + α*Σ XᵢYᵢ` (the other products) is
  compressed once (see [`LeafOperator`](@ref)).
"""
function flush!(d::LazyDriver, C::HMatrix)
    updates = pop!(d.pending, C, nothing)
    isnothing(updates) && return C
    if !isleaf(C)
        _flush_internal_node!(d, C, updates)
        return C
    end
    reads = Iterators.flatten(((Z for (X, Y) in updates.products for Z in (X, Y)), (t.term for t in updates.carry)))
    spawn!(d.executor, reads, (C,)) do
        if isadmissible(C)
            with_buffer(buf -> _flush_low_rank_leaf!(C, updates, d.α, d.comp, buf), d)
        else
            _flush_dense_leaf!(C, updates, d.α)
        end
    end
    return C
end

"""
    flush_tree!(d::LazyDriver, C::HMatrix)

Flush `C` and all of its descendants, top-down.
"""
function flush_tree!(d::LazyDriver, C::HMatrix)
    flush!(d, C)
    foreach(chd -> flush_tree!(d, chd), children(C))
    return C
end

"""
    _flush_internal_node!(d::LazyDriver, C::HMatrix, updates::PendingUpdates)

The flush of the internal node `C` (see [`flush!`](@ref)). With `d.uplo = 'U'` (or
`'L'`), a diagonal block only hands its updates to the children of its upper (or lower)
triangle.
"""
function _flush_internal_node!(d::LazyDriver{S, T}, C::HMatrix, updates) where {S, T}
    @assert !hasdata(C) "internal node with data found"
    # the products with an exact low-rank form, computed by one piece of work, and
    # shared by the leaves below `C`
    low_rank_products = filter(((X, Y),) -> _has_low_rank_form(X, Y), updates.products)
    if !isempty(low_rank_products)
        terms = [LazyLowRankTerm{T}() for _ in low_rank_products]
        append!(updates.carry, CarryTerm(term, 0, 0) for term in terms)
        reads = (Z for (X, Y) in low_rank_products for Z in (X, Y))
        spawn!(d.executor, reads, terms) do
            with_buffer(d) do buf
                for ((X, Y), term) in zip(low_rank_products, terms)
                    term.A, term.B = _compute_low_rank_term(X, Y, d.α, buf.scratch)
                end
            end
        end
    end
    # the other products and the carry are handed to the children
    chdC = children(C)
    diagonal = rowrange(C) == colrange(C)
    skip(i, j) = diagonal && (d.uplo == 'U' ? i > j : d.uplo == 'L' ? i < j : false)
    for (X, Y) in updates.products
        _has_low_rank_form(X, Y) && continue
        @assert !isleaf(X) && !isleaf(Y) "the factors of a product are not clustered like its target"
        chdX, chdY = children(X), children(Y)
        for i in axes(chdC, 1), j in axes(chdC, 2), k in axes(chdX, 2)
            skip(i, j) && continue
            push!(d, chdC[i, j], chdX[i, k], chdY[k, j])
        end
    end
    isempty(updates.carry) && return C
    for j in axes(chdC, 2), i in axes(chdC, 1)
        skip(i, j) && continue
        chd = chdC[i, j]
        row_shift = first(rowrange(chd)) - first(rowrange(C))
        col_shift = first(colrange(chd)) - first(colrange(C))
        carry = pending!(d, chd).carry
        for t in updates.carry
            push!(carry, CarryTerm(t.term, t.row_offset + row_shift, t.col_offset + col_shift))
        end
    end
    return C
end

"""
    _flush_dense_leaf!(C::HMatrix, updates::PendingUpdates, α)

Add the carry and the products `α*X*Y` to the dense data of the leaf `C`.
"""
function _flush_dense_leaf!(C::HMatrix, updates, α)
    d = data(C)::Matrix{eltype(C)}
    m, n = size(C)
    for t in updates.carry
        A, B = factors(t, m, n)
        mul!(d, A, B', true, true)
    end
    for (X, Y) in updates.products
        _mul_dense!(d, X, Y, α)
    end
    return C
end

"""
    _flush_low_rank_leaf!(C::HMatrix, updates::PendingUpdates, α, comp, buf::UpdateBuffer)

Compress `data(C) + Σ AₖBₖ' + α*Σ XᵢYᵢ` for the admissible leaf `C`, with the carry
`(Aₖ, Bₖ)` and the pending products `(Xᵢ, Yᵢ)`.

The low-rank terms (the data, the carry, and the products with an exact low-rank
form, evaluated here) are packed side by side into a single pair of factors `(A, B)`
stored in `buf.lowrank`, so that the compressor samples them with one
matrix-vector product, without allocating. The other products are only sampled by
the compressor, through a [`LeafOperator`](@ref).
"""
function _flush_low_rank_leaf!(C::HMatrix, updates, α, comp, buf::UpdateBuffer)
    T = eltype(C)
    m, n = size(C)
    R = data(C)::RkMatrix{T}
    exact = filter(((X, Y),) -> _has_low_rank_form(X, Y), updates.products)
    implicit = filter(((X, Y),) -> !_has_low_rank_form(X, Y), updates.products)
    r = rank(R) + sum(t -> size(t.term.A, 2), updates.carry; init = 0) +
        sum(((X, Y),) -> _low_rank_term_rank(X, Y), exact; init = 0)
    w = buf.lowrank
    length(w) < (m + n) * r && resize!(w, (m + n) * r)
    A = reshape(view(w, 1:(m * r)), m, r)
    B = reshape(view(w, (m * r + 1):((m + n) * r)), n, r)
    k = 0
    for (Aₜ, Bₜ) in Iterators.flatten((((R.A, R.B),), (factors(t, m, n) for t in updates.carry)))
        cols = (k + 1):(k + size(Aₜ, 2))
        copyto!(view(A, :, cols), Aₜ)
        copyto!(view(B, :, cols), Bₜ)
        k = last(cols)
    end
    for (X, Y) in exact
        cols = (k + 1):(k + _low_rank_term_rank(X, Y))
        _compute_low_rank_term!(view(A, :, cols), view(B, :, cols), X, Y, α, buf.scratch)
        k = last(cols)
    end
    L = LeafOperator{T}(A, B, implicit, α, buf.scratch, buf.rkscratch)
    setdata!(C, comp(L, axes(L, 1), axes(L, 2), buf.comp))
    return C
end

"""
    _is_low_rank(X)

Whether `X` is a low-rank (admissible) leaf.
"""
_is_low_rank(X) = isleaf(X) && isadmissible(X)

"""
    _rank_of(X)

The rank of the low-rank leaf `X` (an `HMatrix` or its adjoint).
"""
_rank_of(X) = size(first(_rkfactors(data(X))), 2)

"""
    _has_low_rank_form(X, Y)

Whether `X*Y` has an exact low-rank form computed by `_compute_low_rank_term`: `X`
or `Y` is a low-rank leaf, or both are dense leaves. This is always the case for a
product pending at an internal node with a leaf factor, since the cluster shared by
`X` and `Y` is then a leaf.
"""
_has_low_rank_form(X, Y) = _is_low_rank(X) || _is_low_rank(Y) || (isleaf(X) && isleaf(Y))

"""
    _through_x(X, Y)

Whether `_compute_low_rank_term` factors `X*Y` through the low-rank leaf `X` (rather
than `Y`): `X` is a low-rank leaf, and `Y` is not one of smaller rank.
"""
_through_x(X, Y) = _is_low_rank(X) && !(_is_low_rank(Y) && _rank_of(Y) < _rank_of(X))

"""
    _low_rank_term_rank(X, Y)

The rank of the factors of `X*Y` computed by `_compute_low_rank_term`.
"""
_low_rank_term_rank(X, Y) = _through_x(X, Y) ? _rank_of(X) : _is_low_rank(Y) ? _rank_of(Y) : size(X, 2)

"""
    _compute_low_rank_term(X, Y, α[, buf])

Return factors `(A, B)` (matrices) such that `α*X*Y = A*B'`, where `X` or `Y` (an
`HMatrix` or its adjoint) is a low-rank leaf, or both are dense leaves (see
`_has_low_rank_form`). The factorization is exact: its rank (see
`_low_rank_term_rank`) is either the smallest rank of the low-rank factors, or the
size of the cluster shared by `X` and `Y`, which is a leaf when `X*Y` targets an
internal node. A factor of a low-rank leaf, or the dense data of `X`, is shared, not
copied.
"""
function _compute_low_rank_term(X, Y, α, buf = eltype(X)[])
    T, r = eltype(X), _low_rank_term_rank(X, Y)
    # share the factor which `_compute_low_rank_term!` copies, and allocate the other
    A, B = if _through_x(X, Y)
        first(_rkfactors(data(X))), Matrix{T}(undef, size(Y, 2), r)
    elseif _is_low_rank(Y)
        Matrix{T}(undef, size(X, 1), r), last(_rkfactors(data(Y)))
    else
        d = data(X)
        (d isa Matrix{T} ? d : Matrix{T}(undef, size(X, 1), r)), Matrix{T}(undef, size(Y, 2), r)
    end
    return _compute_low_rank_term!(A, B, X, Y, α, buf)
end

"""
    _compute_low_rank_term!(A, B, X, Y, α, buf)

Like `_compute_low_rank_term`, but writing the factors into `A` and `B` (which may
already be the factors shared with `X` or `Y`), using the scratch vector `buf`.
"""
function _compute_low_rank_term!(A, B, X, Y, α, buf)
    if _through_x(X, Y) # α*X*Y = U*(conj(α)*Y'*V)'
        U, V = _rkfactors(data(X))
        A === U || copyto!(A, U)
        _mul131!(fill!(B, zero(eltype(B))), adjoint(Y), V, conj(α), buf)
    elseif _is_low_rank(Y) # α*X*Y = (α*X*U)*V'
        U, V = _rkfactors(data(Y))
        _mul131!(fill!(A, zero(eltype(A))), X, U, α, buf)
        B === V || copyto!(B, V)
    else # both dense
        @assert isleaf(X) && isleaf(Y)
        A === data(X) || copyto!(A, data(X))
        B .= conj(α) .* adjoint(data(Y))
    end
    return A, B
end

############################################################################################
# Factorizations and triangular solves
############################################################################################

"""
    lu!(M::HMatrix, compressor; threads = true, executor)

Hierarchical LU factorization of `M` in place, every update being deferred: each
admissible leaf of the factors is compressed exactly once, with `compressor` (e.g.
[`RandomizedSVD`](@ref) or [`PartialACA`](@ref)), from its Schur complement. The work
is parallelized with `threads = true` (when Julia runs on several threads), or run by
the given `executor` ([`ThreadsExecutor`](@ref) or [`SerialExecutor`](@ref)).

`F \\ b` solves in the ordering of the points used to assemble `M`. The factors `F.L`
and `F.U` are stored in the ordering of the cluster trees, with
`F.L * (F.U * x) ≈ Matrix(M)[F.p, F.q] * x`; products and solves with them act in that ordering.

The entries of `M` must be numbers. With `threads = true`, BLAS should run on a single
thread (`BLAS.set_num_threads(1)`), since the parallelism comes from the blocks. An error
raised by a threaded factorization is rethrown without the backtrace of the task which
raised it; `threads = false` gives the full backtrace.
"""
function LinearAlgebra.lu!(M::HMatrix, compressor; threads = true, executor = _executor(threads))
    d = LazyDriver(M, M, M, compressor; executor)
    _run!(() -> _lu!(d, M), executor)
    @assert isempty(d.pending) "updates left after the factorization"
    return LU(M, Int[], 0)
end

"""
    lu!(M::HMatrix; atol, rank, rtol, block, threads = true)

Hierarchical LU factorization of `M`, using the `RandomizedSVD(; atol, rank, rtol, block)`
compressor (see [`RandomizedSVD`](@ref)).
"""
function LinearAlgebra.lu!(M::HMatrix; threads = true, kwargs...)
    return lu!(M, RandomizedSVD(; kwargs...); threads)
end

"""
    LinearAlgebra.lu(M::HMatrix,args...;kwargs...)

Hierarchical LU factorization. See [`lu!`](@ref) for the available options.
"""
LinearAlgebra.lu(M::HMatrix, args...; kwargs...) = lu!(deepcopy(M), args...; kwargs...)

"""
    _lu!(d::LazyDriver, M::HMatrix)

Driver of [`lu!`](@ref) for the diagonal block `M`: flush `M`, then factor it (dense
LU of a leaf), or recurse on its children, pushing the Schur complement updates.
"""
function _lu!(d, M::HMatrix)
    flush!(d, M)
    if isleaf(M)
        spawn!(() -> lu!(data(M)::Matrix{eltype(M)}, NoPivot()), d.executor, (), (M,))
        return M
    end
    chdM = children(M)
    n = size(chdM, 1)
    for i in 1:n
        _lu!(d, chdM[i, i])
        for j in (i + 1):n
            _solve_lower!(d, UnitLowerTriangular(chdM[i, i]), chdM[i, j])
            _solve_upper!(d, chdM[j, i], UpperTriangular(chdM[i, i]))
        end
        for j in (i + 1):n, k in (i + 1):n
            push!(d, chdM[j, k], chdM[j, i], chdM[i, k])
        end
    end
    return M
end

"""
    cholesky!(M::HermitianHMatrix, compressor; threads = true, executor)

Hierarchical Cholesky factorization `M = U'U` in place, the factor being stored in
the upper triangle of `M`, which must hold the data of `M` (as for the `Hermitian`
assembled by [`assemble_hmatrix`](@ref)); an `ArgumentError` is thrown otherwise. As for
[`lu!`](@ref), every update is deferred, each admissible leaf of the factor is
compressed once, and the work is parallelized with `threads = true` or run by the given
`executor`.

`F \\ b` solves in the ordering of the points used to assemble `M`. The factors `F.L`
and `F.U` are stored in the ordering of the cluster tree, with
`F.L * (F.U * x) ≈ Matrix(M)[F.p, F.p] * x`; products and solves with them act in that ordering.
"""
function LinearAlgebra.cholesky!(M::HermitianHMatrix, compressor; threads = true, executor = _executor(threads))
    M.uplo == 'U' || throw(ArgumentError("the data of `M` must be stored in its upper triangle"))
    H = parent(M)
    d = LazyDriver(H, adjoint(H), H, compressor; uplo = 'U', executor)
    _run!(() -> _cholesky!(d, H), executor)
    @assert isempty(d.pending) "updates left after the factorization"
    return M |> UpperTriangular |> Cholesky
end

"""
    cholesky!(M::HermitianHMatrix; atol, rank, rtol, block, threads = true)

Hierarchical Cholesky factorization of `M`, using the
`RandomizedSVD(; atol, rank, rtol, block)` compressor (see [`RandomizedSVD`](@ref)).
"""
function LinearAlgebra.cholesky!(M::HermitianHMatrix; threads = true, kwargs...)
    return cholesky!(M, RandomizedSVD(; kwargs...); threads)
end

"""
    cholesky(M::HermitianHMatrix, args...; kwargs...)

Hierarchical Cholesky factorization. See [`cholesky!`](@ref) for the available
options.
"""
function LinearAlgebra.cholesky(M::HermitianHMatrix, args...; kwargs...)
    return cholesky!(deepcopy(M), args...; kwargs...)
end

"""
    _cholesky!(d::LazyDriver, M::HMatrix)

Driver of [`cholesky!`](@ref) for the diagonal block `M` (whose upper triangle holds
the data): flush `M`, then factor it (dense Cholesky of a leaf), or recurse on its
children, solving for the blocks of the first block row and pushing the updates
`M[j,k] ← M[j,k] - M[i,j]'*M[i,k]` of the upper triangle.
"""
function _cholesky!(d, M::HMatrix)
    flush!(d, M)
    if isleaf(M)
        spawn!(() -> cholesky!(Hermitian(data(M)::Matrix{eltype(M)}, :U), NoPivot()), d.executor, (), (M,))
        return M
    end
    chdM = children(M)
    n = size(chdM, 1)
    for i in 1:n
        _cholesky!(d, chdM[i, i])
        for j in (i + 1):n
            _solve_lower!(d, adjoint(UpperTriangular(chdM[i, i])), chdM[i, j])
        end
        for j in (i + 1):n, k in j:n
            push!(d, chdM[j, k], adjoint(chdM[i, j]), chdM[i, k])
        end
    end
    return M
end

"""
    _solve_lower!(d::LazyDriver, L::HLowerTriangular, X::HMatrix)

`X ← L⁻¹X`: flush `X`, then solve with its data (a leaf), or recurse on its children
by block forward substitution, pushing the updates `X[j,k] ← X[j,k] - L[j,i]*X[i,k]`
(the coefficient of `d` is `-1`). `L` is a triangle of an `HMatrix` or of its adjoint
(e.g. `UnitLowerTriangular(D)` for an LU, `UpperTriangular(D)'` for a Cholesky
factorization), whose blocks are final.
"""
function _solve_lower!(d, L::HLowerTriangular, X::HMatrix)
    flush!(d, X)
    if isleaf(X)
        spawn!(() -> ldiv!(L, data(X)), d.executor, (L,), (X,))
        return X
    end
    isleaf(parent(L)) && throw(ArgumentError("the triangular factor is a leaf but the right-hand side is not"))
    chdL, chdX = children(parent(L)), children(X)
    n = size(chdL, 1)
    for k in axes(chdX, 2), i in 1:n
        _solve_lower!(d, _wraptriangular(chdL[i, i], L), chdX[i, k])
        for j in (i + 1):n
            push!(d, chdX[j, k], chdL[j, i], chdX[i, k])
        end
    end
    return X
end

"""
    _solve_upper!(d::LazyDriver, X::HMatrix, U::HUpperTriangular)

`X ← XU⁻¹`; see `_solve_lower!`.
"""
function _solve_upper!(d, X::HMatrix, U::HUpperTriangular)
    flush!(d, X)
    if isleaf(X)
        spawn!(d.executor, (U,), (X,)) do
            R = data(X)
            R isa RkMatrix ? with_buffer(buf -> rdiv!(R, U, buf.scratch), d) : rdiv!(R, U)
        end
        return X
    end
    isleaf(parent(U)) && throw(ArgumentError("the triangular factor is a leaf but the right-hand side is not"))
    chdU, chdX = children(parent(U)), children(X)
    n = size(chdU, 2)
    for k in axes(chdX, 1), i in 1:n
        _solve_upper!(d, chdX[k, i], _wraptriangular(chdU[i, i], U))
        for j in (i + 1):n
            push!(d, chdX[k, j], chdX[k, i], chdU[i, j])
        end
    end
    return X
end
