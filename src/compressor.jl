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
    irange = index_range(rowtree)
    jrange = index_range(coltree)
    aca(K,irange,jrange)
end

function (aca::ACA)(K,irange::UnitRange,jrange::UnitRange)
    M  = K[irange,jrange] # computes the entire matrix.
    _aca_full!(M,aca.atol,aca.rank,aca.rtol)
end

function _aca_full!(M, atol, rmax, rtol)
    Madj  = adjoint(M)
    T   = eltype(M)
    m,n = size(M)
    A   = Vector{Vector{T}}()
    B   = Vector{Vector{T}}()
    er = Inf
    exact_norm = norm(M,2) # exact norm
    r = 0 # current rank
    while er > max(atol,rtol*exact_norm) && r < rmax
        I = _aca_full_pivot(M)
        i,j = Tuple(I)
        δ   = M[I]
        if svdvals(δ)[end] == 0
            return RkMatrix(A,B)
        else
            iδ = inv(δ)
            col  = M[:,j]
            row  = Madj[:,i]
            for k in eachindex(row)
                row[k] = row[k]*adjoint(iδ)
            end
            # for k in eachindex(col)
            #     col[k] = col[k]*iδ
            # end
            r += 1
            push!(A,col)
            push!(B,row)
            axpy!(-1,col*adjoint(row),M) # M <-- M - col*row'
            # axpy!(-1,col*reshape(row,1,n),M) # M <-- M - col*row'
            er = norm(M,2) # exact error
            # @show r,er
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
    irange = index_range(rowtree)
    jrange = index_range(coltree)
    _aca_partial(K,irange,jrange,paca.atol,paca.rank,paca.rtol,istart-irange.start+1)
end

function _aca_partial_initial_pivot(rowtree)
    # return index_range(rowtree).start
    # the method below is suggested in Bebendorf, but it does not seem to
    # improve the  error. The idea is that the initial pivot is the closes point
    # to the center of the container.
    xc        = center(container(rowtree))
    d         = Inf
    els       = root_elements(rowtree)
    loc_idxs  = index_range(rowtree)
    istart    = first(loc_idxs)
    for i in loc_idxs
        x     = els[i]
        if norm(x-xc) < d
            d      = norm(x-xc)
            istart = i
        end
    end
    return istart
end

function (paca::PartialACA)(K,irange::UnitRange,jrange::UnitRange)
    _aca_partial(K,irange,jrange,paca.atol,paca.rank,paca.rtol)
end

function _aca_partial(K,irange,jrange,atol,rmax,rtol,istart=1)
    Kadj = adjoint(K)
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
        # compute next row by row <-- K[i+ishift,jrange] - R[i,:]
        row    = Kadj[jrange,i+ishift]
        for k = 1:r
            axpy!(-adjoint(A[k][i]),B[k],row)
            # for j in eachindex(row)
            #     row[j] = row[j] - B[k][j]*adjoint(A[k][i])
            # end
        end
        j    = _aca_partial_pivot(row,J)
        δ    = row[j]
        if svdvals(δ)[end] == 0
            i = findfirst(x->x==true,J)
        else # δ != 0
            iδ = inv(δ)
            # rdiv!(b,δ) # b <-- b/δ
            for k in eachindex(row)
                row[k] = row[k]*iδ
            end
            J[j] = false
            # compute next col by col <-- K[irange,j+jshift] - R[:,j]
            col    = K[irange,j+jshift]
            for k = 1:r
                axpy!(-adjoint(B[k][j]),A[k],col)
                # for i in eachindex(col)
                #     col[i] = col[i] - A[k][i]*adjoint(B[k][j])
                # end
            end
            # push new cross and increase rank
            r += 1
            push!(A,col)
            push!(B,row)
            # estimate the error by || R_{k} - R_{k-1} || = ||a|| ||b||
            er       = norm(col)*norm(row)
            # estimate the norm by || K || ≈ || R_k ||
            est_norm = _update_frob_norm(est_norm,A,B)
            i        = _aca_partial_pivot(col,I)
            # @show r, er
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
            out += 2*real(dot(A[l],a)*(dot(b,B[l])))
        end
    end
    return sqrt(cur^2 + out)
end

"""
    _aca_partial_pivot(v,I)

Find the index of the element `x ∈ v` maximizing its smallest singular value.
This is equivalent to minimizing the spectral norm of the inverse of `x`.

When `x` is a scalar, this is simply the element with largest absolute value.

See
(https://www.sciencedirect.com/science/article/pii/S0021999117306721)[https://www.sciencedirect.com/science/article/pii/S0021999117306721]
more details.
"""
function _aca_partial_pivot(v,J)
    idx = -1
    val = -Inf
    for n in 1:length(J)
        J[n] || continue
        x   = v[n]
        σ   = svdvals(x)[end]
        σ < val && continue
        idx = n
        val = σ
    end
    return idx
end

function _aca_full_pivot(M)
    idxs = CartesianIndices(M)
    idx  = first(idxs)
    val = -Inf
    for I in idxs
        x   = M[I]
        σ   = svdvals(x)[end]
        σ < val && continue
        idx = I
        val = σ
    end
    return idx
end



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
    irange = index_range(rowtree)
    jrange = index_range(coltree)
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
        B = @views (F.V[:,1:r])*Diagonal(F.S[1:r])
    end
    return RkMatrix(A,B)
end
