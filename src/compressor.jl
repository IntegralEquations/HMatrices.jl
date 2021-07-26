"""
    struct ACA

Adaptive cross approximation algorithm with full pivoting. Requires evaluation
of entire matrix to be compressed, but is guaranteed to work.

`ACA` objects are used as functors through the following syntax:
  `aca(K,irange::UnitRange,jrange::UnitRange)`. This produces an approximation
  of the matrix `K[irange,jrange]`, where `K` is a matrix-like object with
  `getindex(K,i,j)` implemented.
"""
@Base.kwdef struct ACA
    atol::Float64 = 0
    rank::Int     = typemax(Int)
    rtol::Float64 = atol>0 || rank<typemax(Int) ? 0 : sqrt(eps(Float64))
end

function (aca::ACA)(K,rowtree::ClusterTree,coltree::ClusterTree)
    irange = range(rowtree)
    jrange = range(coltree)
    aca(K,irange,jrange)
end

function (aca::ACA)(K,irange::UnitRange,jrange::UnitRange)
    M  = K[irange,jrange] # computes the entire matrix.
    _aca_full!(M,aca.atol,aca.rank,aca.rtol)
end

function _aca_full!(M, atol, rmax, rtol)
    T   = eltype(M)
    m,n = size(M)
    A   = Vector{Vector{T}}()
    B   = Vector{Vector{T}}()
    er = Inf
    exact_norm = norm(M,2) # exact norm
    r = 0 # current rank
    while er > max(atol,rtol*exact_norm) && r < rmax
        (i,j) = argmax(abs.(M)).I
        δ       = M[i,j]
        if δ == 0
            return RkMatrix(A,B)
        else
            a = M[:,j]
            b = conj(M[i,:])
            rdiv!(a,δ)
            r += 1
            push!(A,a)
            push!(B,b)
            axpy!(-1,a*adjoint(b),M) # M <-- M - col*row'
            er = norm(M,2) # exact error
        end
    end
    return RkMatrix(A,B)
end

"""
    struct PartialACA

Adaptive cross approximation algorithm with partial pivoting. Does not require
evaluation of the entire matrix to be compressed, but is not guaranteed to converge either.
"""
@Base.kwdef struct PartialACA
    atol::Float64 = 0
    rank::Int     = typemax(Int)
    rtol::Float64 = atol>0 || rank<typemax(Int) ? 0 : sqrt(eps(Float64))
end

function (paca::PartialACA)(K,rowtree::ClusterTree,coltree::ClusterTree)
    # find initial column pivot for partial ACA
    istart = _aca_partial_initial_pivot(rowtree)
    irange = range(rowtree)
    jrange = range(coltree)
    _aca_partial(K,irange,jrange,paca.atol,paca.rank,paca.rtol,istart-irange.start+1)
end

function _aca_partial_initial_pivot(rowtree)
    # return range(rowtree).start
    # the method below is suggested in Bebendorf, but it does not seem to
    # improve the  error
    xc        = center(rowtree.bounding_box)
    d         = Inf
    pts       = rowtree.points
    loc_idxs  = rowtree.loc_idxs
    istart    = first(loc_idxs)
    for i in loc_idxs
        iglob = rowtree.loc2glob[i]
        x     = pts[iglob]
        if norm(x-xc) < d
            d = norm(x-xc)
            istart = i
        end
    end
    return istart
end

function (paca::PartialACA)(K,irange::UnitRange,jrange::UnitRange)
    _aca_partial(K,irange,jrange,paca.atol,paca.rank,paca.rtol)
end

function _aca_partial(K,irange,jrange,atol,rmax,rtol,istart=1)
    rmax = min(length(irange),length(jrange),rmax)
    ishift,jshift = irange.start-1, jrange.start-1 # maps global indices to local indices
    T   = Base.eltype(K)
    m,n = length(irange),length(jrange)
    A   = Vector{Vector{T}}()
    B   = Vector{Vector{T}}()
    I   = BitVector(true for i = 1:m)
    J   = BitVector(true for i = 1:n)
    i   = istart # initial pivot
    er  = Inf
    est_norm = 0 # approximate norm of K[irange,jrange]
    r   = 0 # current rank
    while er > max(atol,rtol*est_norm) && r < rmax
        # remove index i from allowed row
        I[i] = false
        # compute next row by b <-- conj(K[i+ishift,jrange] - R[i,:])
        b    = conj!(K[i+ishift,jrange])
        for k = 1:r
            axpy!(-conj(A[k][i]),B[k],b)
        end
        j    = _nextcol(b,J)
        δ    = b[j]
        if δ == 0
            i = findfirst(x->x==true,J)
        else # δ != 0
            rdiv!(b,δ) # b <-- b/δ
            J[j] = false
            # compute next col by a <-- K[irange,j+jshift] - R[:,j]
            a    = K[irange,j+jshift]
            for k = 1:r
                axpy!(-conj(B[k][j]),A[k],a)
            end
            # push new cross and increase rank
            r += 1
            push!(A,a)
            push!(B,b)
            # estimate the error by || R_{k} - R_{k-1} || = ||a|| ||b||
            er       = norm(a)*norm(b)
            # estimate the norm by || K || ≈ || R_k ||
            est_norm = _update_frob_norm(est_norm,A,B)
            i        = _nextrow(a,I)
        end
    end
    return RkMatrix(A,B)
end

"""
    _update_frob_norm(acc,A,B)

Given the Frobenius norm of `Rₖ = A[1:end-1]*adjoint(B[1:end-1])` in `acc`, compute the
Frobenius norm of `Rₖ₊₁ = A*adjoint(B)` efficiently.
"""
@inline function _update_frob_norm(cur,A,B)
    @timeit_debug "Update Frobenius norm" begin
        k = length(A)
        a = A[end]
        b = B[end]
        out = norm(a)^2 * norm(b)^2
        for l=1:k-1
            out += 2*real(dot(A[l],a)*conj(dot(B[l],b)))
        end
    end
    return sqrt(cur^2 + out)
end

"""
    _nextcol(col,J)
    _nextrow(row,I)
Find the entry in `col` (resp. `row`) with largest absolute value within the
valid set `J` (resp `I`).
"""
function _nextcol(col,J)
    out = -1
    val = -Inf
    for n in 1:length(J)
        J[n] || continue
        tmp = abs(col[n])
        tmp < val && continue
        out = n
        val = tmp
    end
    return out
end
_nextrow(row,I) = _nextcol(row,I)

"""
    struct TSVD

Compression algorithm based on *a posteriori* truncation of an `SVD`. This is
the optimal approximation in Frobenius norm; however, it also tends to be very
expensive and thus should be used mostly for "small" matrices.
"""
@Base.kwdef struct TSVD
    atol::Float64 = 0
    rank::Int     = typemax(Int)
    rtol::Float64 = atol>0 || rank<typemax(Int) ? 0 : sqrt(eps(Float64))
end

function (tsvd::TSVD)(K,rowtree::ClusterTree,coltree::ClusterTree)
    irange = range(rowtree)
    jrange = range(coltree)
    tsvd(K,irange,jrange)
end

function (tsvd::TSVD)(K,irange::UnitRange,jrange::UnitRange)
    M = K[irange,jrange]
    compress!(M,tsvd)
end

function compress!(M::Matrix{T},tsvd::TSVD) where {T}
    F     = svd!(M)
    enorm = F.S[1]
    r     = findlast(x -> x>max(tsvd.atol,tsvd.rtol*enorm), F.S) + 1
    r     = min(r,tsvd.rank)
    m,n = size(M)
    if m<n
        A = @views F.U[:,1:r]*Diagonal(F.S[1:r])
        B = F.V[:,1:r]
    else
        A = F.U[:,1:r]
        B = @views F.V[:,1:r]*adjoint(Diagonal(F.S[1:r]))
    end
    return RkMatrix(A,B)
end
