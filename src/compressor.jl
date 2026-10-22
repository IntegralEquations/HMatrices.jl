"""
    abstract type AbstractCompressor

Types used to compress matrices. A compressor `comp` is called as `comp(K,
irange, jrange, buf)` and returns an [`RkMatrix`](@ref) approximating
`K[irange, jrange]`, where `buf` is a workspace created by
[`allocate_buffer`](@ref)`(comp, T)` (`nothing` by default). During the assembly of an
[`HMatrix`](@ref), it is called with the index ranges of each admissible block.
"""
abstract type AbstractCompressor end

"""
    struct PartialACA

Adaptive cross approximation algorithm with partial pivoting. This structure can be
used to generate an [`RkMatrix`](@ref) from a matrix-like object `M` as follows:

```jldoctest
using LinearAlgebra
using HMatrices: PartialACA
rtol = 1e-6
comp = PartialACA(;rtol)
A = rand(10,2)
B = rand(10,2)
M = A*adjoint(B) # a low-rank matrix
R = comp(M, axes(M)...) # compress the entire matrix `M`
norm(Matrix(R) - M) < rtol*norm(M) # true

# output

true

```

Because it uses partial pivoting, the linear operator does not have to be
evaluated at every `i,j`. This is usually much faster than [`TSVD`](@ref), but
due to the pivoting strategy the algorithm may fail in special cases, even when
the underlying linear operator is of low rank.

The default `rtol` is `sqrt(eps(R))`, where `R` is the real type underlying the
entries of the compressed matrix, or only the accuracy attainable with `R` if `atol` or
`rank` is given. An `rtol` smaller than that accuracy (`10eps(R)`) is raised to it, or
throws an `ArgumentError` if it is positive, so the rank of a block can stay below
`rank` when the rest of the block is at the level of rounding errors.
"""
Base.@kwdef struct PartialACA <: AbstractCompressor
    atol::Float64 = 0
    rank::Int = typemax(Int)
    rtol::Union{Nothing, Float64} = nothing
    PartialACA(atol, rank, rtol) = new(_check_tolerances(atol, rank, rtol)...)
end

function (paca::PartialACA)(K, rowtree::ClusterTree, coltree::ClusterTree, bufs = nothing)
    # find initial column pivot for partial ACA
    istart = _aca_partial_initial_pivot(rowtree)
    irange = index_range(rowtree)
    jrange = index_range(coltree)
    return _aca_partial(
        K,
        irange,
        jrange,
        paca.atol,
        paca.rank,
        paca.rtol,
        istart - irange.start + 1,
        bufs,
    )
end

function (paca::PartialACA)(K, irange::AbstractRange, jrange::AbstractRange, bufs = nothing)
    return _aca_partial(K, irange, jrange, paca.atol, paca.rank, paca.rtol, 1, bufs)
end

(paca::PartialACA)(K::AbstractMatrix) = paca(K, axes(K, 1), axes(K, 2))

struct ACABuffer{T}
    A::VectorOfVectors{T}
    B::VectorOfVectors{T}
    I::BitVector
    J::BitVector
end

function ACABuffer(T, m = 0, n = 0)
    return ACABuffer{T}(
        VectorOfVectors(T, m),
        VectorOfVectors(T, n),
        BitVector(undef, m),
        BitVector(undef, n),
    )
end

struct ACAWithRecompressionBuffer{T}
    aca::ACABuffer{T}
    data::Vector{T}   # flat scratch, grown via resize!, never shrunk
end

function ACAWithRecompressionBuffer(T)
    return ACAWithRecompressionBuffer{T}(ACABuffer(T), Vector{T}(undef, 0))
end

function _get_scratch!(buf::ACAWithRecompressionBuffer{T}, k) where {T}
    n = k * k
    if length(buf.data) < n
        resize!(buf.data, n)
    end
    return reshape(view(buf.data, 1:n), k, k)
end

"""
    _aca_partial(K,irange,jrange,atol,rmax,rtol,istart=1)

Internal function implementing the adaptive cross-approximation algorithm with
partial pivoting. The returned `R::RkMatrix` provides an approximation to
`M = K[irange,jrange]` of rank at most `rmax`, which is expected to satisfy `|M - R| <
max(atol,rtol*|M|)`, but this inequality may fail to hold due to the various
errors involved in estimating the error and |M|.
"""
function _aca_partial(K, irange, jrange, atol, rmax, rtol, istart, buffer_ = nothing)
    Kadj = adjoint(K)
    T = eltype(K)
    rtol = _rtol(rtol, atol, rmax, T)
    m, n = length(irange), length(jrange)
    # maps local indices to indices of `K`
    ishift, jshift = first(irange) - 1, first(jrange) - 1
    buffer = isnothing(buffer_) ? ACABuffer(T, m, n) : buffer_
    A, B, I, J = buffer.A, buffer.B, buffer.I, buffer.J # extract the individual buffers for convenience
    # reset rather than assume empty buffers: a previous call may have thrown
    reset!(A)
    reset!(B)
    A.m, B.m = m, n
    rmax = min(m, n, rmax)
    resize!(I, m)
    resize!(J, n)
    fill!(I, true)
    fill!(J, true)
    i = istart # initial pivot
    er = Inf
    est_norm = 0 # approximate norm of K[irange,jrange]
    r = 0 # current rank
    zero_rows = true # whether every rejected row of the residual was exactly zero
    while er > max(atol, rtol * est_norm) && r < rmax
        # remove index i from allowed row
        I[i] = false
        # pre-allocate row and column
        a = newcol!(A)
        A.k -= 1
        b = newcol!(B)
        B.k -= 1
        # compute next row by row <-- K[i+ishift,jrange] - R[i,:]
        getblock!(b, Kadj, jrange, i + ishift)
        for k in 1:r
            axpy!(-adjoint(A[k][i]), B[k], b)
        end
        j = _aca_partial_pivot(b, J)
        δ = b[j]
        σ = min_svd_vals(δ)
        if σ > 0 # compute next col by col <-- K[irange,j+jshift] - R[:,j]
            getblock!(a, K, irange, j + jshift)
            for k in 1:r
                axpy!(-adjoint(B[k][j]), A[k], a)
            end
        end
        # a pivot at the rounding level of its column means that row `i` of the
        # residual is zero (up to rounding errors, e.g. when `K` is a product)
        if σ == 0 || σ <= sqrt(eps(real(eltype(T)))) * maximum(norm, a)
            @debug "zero pivot found during partial aca"
            zero_rows &= iszero(σ)
            i = findfirst(x -> x == true, I)
            if isnothing(i)
                # ran out of candidate rows: every row has been evaluated, so the
                # approximation is exact if the rejected rows were exactly zero on the
                # remaining columns (e.g. a zero block); otherwise some were only small,
                # and ACA may have failed
                zero_rows || @warn "aca possibly failed on $irange × $jrange" maxlog = 1
                break
            end
        else # δ != 0
            iδ = T <: AbstractArray ? pinv(δ; rtol = 1.0e-8) : inv(δ)
            # b <-- b/δ
            for k in eachindex(b)
                b[k] = b[k] * iδ
            end
            J[j] = false
            # increase rank and push cross
            r += 1
            A.k += 1
            B.k += 1
            # estimate the error by || R_{k} - R_{k-1} || = ||a bᴴ||
            er = sqrt(_frob2(a, b))
            # estimate the norm by || K || ≈ || R_k ||
            est_norm = _update_frob_norm(est_norm, A, B)
            i = _aca_partial_pivot(a, I)
            # no candidate row left: every row was used, so the approximation is exact
            i == -1 && break
            @debug r, er, m, n
        end
    end
    # copy from buffer to matrix and create Rk matrix
    R_ = RkMatrix(Matrix(A), Matrix(B))
    # # indicate that `A` and `B` can be reused
    reset!(A)
    reset!(B)
    return R_
end

"""
    _update_frob_norm(acc,A,B)

Given the Frobenius norm of `Rₖ = A[1:end-1]*adjoint(B[1:end-1])` in `acc`,
compute the Frobenius norm of `Rₖ₊₁ = A*adjoint(B)` efficiently.
"""
@inline function _update_frob_norm(cur, A, B)
    k = length(A)
    a = A[k]
    b = B[k]
    out = _frob2(a, b)
    for l in 1:(k - 1)
        out += 2 * real(tr(_gram(A[l], a) * _gram(b, B[l])))
    end
    # rounding errors can make the update negative when `cur` is small
    return sqrt(max(cur^2 + out, zero(out)))
end

"""
    _gram(x, y)

`∑ᵢ xᵢᴴ yᵢ`. For scalar entries this is `dot(x, y)`; for matrix entries it is a
matrix, since the entries do not commute.
"""
_gram(x::AbstractVector{<:Number}, y) = dot(x, y)
function _gram(x, y)
    s = zero(adjoint(first(x)) * first(y))
    for i in eachindex(x, y)
        s += adjoint(x[i]) * y[i]
    end
    return s
end

"""
    _frob2(a, b)

The squared Frobenius norm of the cross `a bᴴ`, i.e. `tr(∑ᵢ aᵢᴴ aᵢ ∑ⱼ bⱼᴴ bⱼ)`.
"""
_frob2(a::AbstractVector{<:Number}, b) = norm(a)^2 * norm(b)^2
_frob2(a, b) = real(tr(_gram(a, a) * _gram(b, b)))

min_svd_vals(x::Number) = abs(x)
function min_svd_vals(A::SMatrix)
    n, m = size(A)
    if n <= 3 && m <= 3
        # the eigenvalues of the smaller Gram matrix are the squared singular values;
        # they have a closed form, much faster than `svdvals`
        G = n <= m ? A * adjoint(A) : adjoint(A) * A
        λ = minimum(eigvals(Hermitian(G)))
        return sqrt(max(λ, zero(λ)))
    else
        return svdvals(A)[end]
    end
end
min_svd_vals(A::Any) = svdvals(A::Any)[end] # fallback

"""
    _aca_partial_pivot(v,I)

Find in the valid set `I` the index of the element `x ∈ v` maximizing its
smallest singular value. This is equivalent to minimizing the spectral norm of
the inverse of `x`.

When `x` is a scalar, this is simply the element with largest absolute value.
Return `-1` if `I` has no valid index.

This general implementation should work for both scalar as well as tensor-valued
kernels; see
[https://www.sciencedirect.com/science/article/pii/S0021999117306721](https://www.sciencedirect.com/science/article/pii/S0021999117306721)
for more details.
"""
function _aca_partial_pivot(v, J)
    idx = -1
    val = -Inf
    for n in 1:length(J)
        J[n] || continue
        σ = min_svd_vals(v[n])
        if σ > val
            idx = n
            val = σ
        end
    end
    return idx
end

"""
    _check_tolerances(atol, rank, rtol)

Throw an `ArgumentError` unless `atol ≥ 0`, `rank ≥ 1` and `rtol` is `nothing` or
`≥ 0`; return them otherwise. Whether `rtol` is attainable depends on the entries, and
is checked by [`_rtol`](@ref).
"""
function _check_tolerances(atol, rank, rtol)
    atol >= 0 || throw(ArgumentError("atol = $atol must be nonnegative"))
    rank >= 1 || throw(ArgumentError("rank = $rank must be positive"))
    isnothing(rtol) || rtol >= 0 || throw(ArgumentError("rtol = $rtol must be nonnegative"))
    return atol, rank, rtol
end

"""
    _rtol(rtol, atol, rank, T)

The relative tolerance used to compress a matrix with entries of type `T`, where `R` is
the real type underlying `T`. If `rtol` is `nothing`, it is `sqrt(eps(R))`, or the
accuracy attainable with `R` (`10eps(R)`) if `atol` or `rank` is given. Throw an
`ArgumentError` if `rtol` is below that accuracy.

An `rtol` of zero is raised to that accuracy, so that an `atol` too small for a block
does not compress it to full rank.
"""
function _rtol(rtol, atol, rank, T)
    R = real(eltype(T))
    if isnothing(rtol)
        return atol > 0 || rank < typemax(Int) ? 10 * eps(R) : sqrt(eps(R))
    end
    # the error of the compressors stagnates at a few `eps(R)`; below that the
    # stopping criterion is never met and the blocks are compressed to full rank
    0 < rtol < 10 * eps(R) && throw(
        ArgumentError(
            "rtol = $rtol is below the accuracy attainable with $R (eps = $(eps(R))); use rtol ≥ $(10 * eps(R))",
        ),
    )
    return max(rtol, 10 * eps(R))
end

function _aca_partial_initial_pivot(rowtree)
    # the method below is suggested in Bebendorf, but it does not seem to
    # improve the error. The idea is that the initial pivot is the closest
    # point to the center of the cluster
    xc = center(container(rowtree))
    d = Inf
    els = root_elements(rowtree)
    loc_idxs = index_range(rowtree)
    istart = first(loc_idxs)
    for i in loc_idxs
        x = els[i]
        if norm(x - xc) < d
            d = norm(x - xc)
            istart = i
        end
    end
    return istart
end

"""
    struct TSVD

Compression algorithm based on *a posteriori* truncation of an `SVD`: the singular
values `σₖ ≤ max(atol, rtol*σ₁)` are dropped, and at most `rank` are kept. This is the
best approximation of its rank, and its error in the spectral norm is below the
tolerance; note that [`PartialACA`](@ref) estimates its error in the Frobenius norm
instead. It evaluates every entry of the block, so it should be used mostly for "small"
matrices. The default `rtol` is the same as for [`PartialACA`](@ref).
"""
Base.@kwdef struct TSVD <: AbstractCompressor
    atol::Float64 = 0
    rank::Int = typemax(Int)
    rtol::Union{Nothing, Float64} = nothing
    TSVD(atol, rank, rtol) = new(_check_tolerances(atol, rank, rtol)...)
end

function (tsvd::TSVD)(K, irange::AbstractRange, jrange::AbstractRange, buf = nothing)
    M = K[irange, jrange]
    return compress!(M, tsvd)
end

(comp::TSVD)(K::Base.Matrix) = comp(K, 1:size(K, 1), 1:size(K, 2))

# number of singular values `S` (sorted decreasingly) above the tolerance, at most
# `rmax` (zero if they are all below the tolerance)
function _truncation_rank(S, atol, rtol, rmax)
    rtol = _rtol(rtol, atol, rmax, eltype(S))
    isempty(S) && return 0
    tol = max(atol, rtol * S[1])
    return min(count(>(tol), S), rmax)
end

"""
    compress!(M::RkMatrix,tsvd::TSVD)

Recompress the matrix `R` using a truncated svd of `R`. The implementation uses
the `qr-svd` strategy to efficiently compute `svd(R)` when `rank(R) ≪
min(size(R))`.
"""
compress!(R::RkMatrix, tsvd::TSVD) = _recompress!(R, tsvd.atol, tsvd.rtol, tsvd.rank)

"""
    compress!(M::Matrix,tsvd::TSVD)

Recompress the matrix `M` using a truncated svd and output an `RkMatrix`. The
data in `M` is invalidated in the process.
"""
function compress!(M::Base.Matrix, tsvd::TSVD)
    m, n = size(M)
    F = svd!(M)
    r = _truncation_rank(F.S, tsvd.atol, tsvd.rtol, tsvd.rank)
    if m < n
        A = @views F.U[:, 1:r] * Diagonal(F.S[1:r])
        B = F.V[:, 1:r]
    else
        A = F.U[:, 1:r]
        B = @views (F.V[:, 1:r]) * Diagonal(F.S[1:r])
    end
    return RkMatrix(A, B)
end

"""
    struct ACAWithRecompression

    ACAWithRecompression(; atol, rank, rtol)

A two-stage compressor that applies [`PartialACA`](@ref) followed by
recompression via [`TSVD`](@ref), both with the tolerances `atol`, `rank` and `rtol`.
The ACA step produces a low-rank approximation quickly using partial pivoting; the
subsequent SVD step can reduce the rank further, at the cost of a small additional
computation. The errors of the two steps add up, so the error can be up to twice the
tolerance.
"""
struct ACAWithRecompression <: AbstractCompressor
    aca::PartialACA
end

ACAWithRecompression(; kwargs...) = ACAWithRecompression(PartialACA(; kwargs...))

function _recompress!(R::RkMatrix{T}, atol, rtol, rmax, bufs = nothing) where {T}
    m, n = size(R)
    k = rank(R)
    QA, RA = qr!(R.A)
    QB, RB = qr!(R.B)
    if isnothing(bufs)
        C = RA * adjoint(RB)
    else
        C = _get_scratch!(bufs, k)
        mul!(C, RA, adjoint(RB))
    end
    F = svd!(C)
    r = _truncation_rank(F.S, atol, rtol, rmax)
    R.A = QA * (F.U[:, 1:r] * Diagonal(F.S[1:r]))
    R.B = QB * F.V[:, 1:r]
    return R
end

# matrix entries: truncate the equivalent scalar matrix, whose rank is rounded up to a
# multiple of the size of the entries
function _recompress!(R::RkMatrix{S}, atol, rtol, rmax, bufs = nothing) where {S <: SMatrix}
    q = size(S, 2)
    Rs = RkMatrix(_scalar_matrix(R.A), _scalar_matrix(R.B))
    _recompress!(Rs, atol, rtol, rmax > typemax(Int) ÷ q ? typemax(Int) : q * rmax)
    R.A = _block_matrix(S, Rs.A)
    R.B = _block_matrix(S, Rs.B)
    return R
end

function compress!(M::Base.Matrix{S}, tsvd::TSVD) where {S <: SMatrix}
    q = size(S, 2)
    rmax = tsvd.rank > typemax(Int) ÷ q ? typemax(Int) : q * tsvd.rank
    Rs = compress!(_scalar_matrix(M), TSVD(tsvd.atol, rmax, tsvd.rtol))
    return RkMatrix(_block_matrix(S, Rs.A), _block_matrix(S, Rs.B))
end

"""
    _scalar_matrix(A::Matrix{<:SMatrix})

The scalar matrix whose blocks are the entries of `A`.
"""
function _scalar_matrix(A::Base.Matrix{S}) where {S <: SMatrix}
    p, q = size(S)
    out = Base.Matrix{eltype(S)}(undef, size(A, 1) * p, size(A, 2) * q)
    for j in axes(A, 2), i in axes(A, 1)
        out[((i - 1) * p + 1):(i * p), ((j - 1) * q + 1):(j * q)] = A[i, j]
    end
    return out
end

"""
    _block_matrix(S, M)

The matrix with entries of type `S` made of the blocks of `M`, whose columns are padded
with zeros to a multiple of the size of `S`.
"""
function _block_matrix(::Type{S}, M) where {S <: SMatrix}
    p, q = size(S)
    m, k = size(M, 1) ÷ p, cld(size(M, 2), q)
    Mp = [M zeros(eltype(M), size(M, 1), k * q - size(M, 2))]
    return [S(view(Mp, ((i - 1) * p + 1):(i * p), ((j - 1) * q + 1):(j * q))) for i in 1:m, j in 1:k]
end

function (comp::ACAWithRecompression)(K, rowtree::ClusterTree, coltree::ClusterTree, bufs = nothing)
    aca_buf = isnothing(bufs) ? nothing : bufs.aca
    # the method of `PartialACA` with trees chooses the first pivot from the geometry
    R = comp.aca(K, rowtree, coltree, aca_buf)
    return _recompress!(R, comp.aca.atol, comp.aca.rtol, comp.aca.rank, bufs)
end

function (comp::ACAWithRecompression)(K, irange::AbstractRange, jrange::AbstractRange, bufs = nothing)
    aca_buf = isnothing(bufs) ? nothing : bufs.aca
    R = comp.aca(K, irange, jrange, aca_buf)
    return _recompress!(R, comp.aca.atol, comp.aca.rtol, comp.aca.rank, bufs)
end

(comp::ACAWithRecompression)(K::AbstractMatrix) = comp(K, axes(K, 1), axes(K, 2))

"""
    _compress(comp, K, rowtree, coltree, buf)

Compress the block of `K` with rows in `rowtree` and columns in `coltree`. Calls
`comp(K, irange, jrange, buf)` with the index ranges of the trees, except for compressors
that use the geometry of the clusters.
"""
function _compress(comp, K, rowtree, coltree, buf)
    return comp(K, index_range(rowtree), index_range(coltree), buf)
end
_compress(comp::Union{PartialACA, ACAWithRecompression}, K, rowtree, coltree, buf) = comp(K, rowtree, coltree, buf)

"""
    _check_rtol(comp, T)

Check that the tolerances of `comp` are attainable for entries of type `T`, before
compressing any block.
"""
_check_rtol(comp, T) = nothing
function _check_rtol(comp::Union{PartialACA, TSVD}, T)
    _rtol(comp.rtol, comp.atol, comp.rank, T)
    return nothing
end
_check_rtol(comp::ACAWithRecompression, T) = _check_rtol(comp.aca, T)

"""
    allocate_buffer(comp, T)

A workspace for the compressor `comp` on matrices with entries of type `T`, which
can be passed to (and reused by) its calls; `nothing` if `comp` needs none.
"""
function allocate_buffer end

allocate_buffer(::PartialACA, T) = ACABuffer(T)
allocate_buffer(::ACAWithRecompression, T) = ACAWithRecompressionBuffer(T)
allocate_buffer(::Any, T) = nothing
