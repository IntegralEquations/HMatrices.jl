# ℋ-matrix/vector product

function _mul_recursive!(C::AbstractVector,A::HMatrix,B::AbstractVector)
    if isleaf(A)
        irange = rowrange(A)
        jrange = colrange(A)
        data   = A.data
        LinearAlgebra.mul!(view(C,irange),data,view(B,jrange))
    else
        for block in A.children
            _mul_recursive!(C,block,B)
        end
    end
    return C
end

function _mul_CPU!(C::AbstractVector,A::HMatrix,B::AbstractVector)
    _mul_recursive!(C,A,B)
    return C
end

function _mul_threads!(C::AbstractVector,A::HMatrix,B::AbstractVector)
    # make copies of C and run in parallel
    nt        = Threads.nthreads()
    Cthreads  = [zero(C) for _ in 1:nt]
    blocks    = getnodes(x -> isleaf(x),A)
    @sync for block in blocks
        Threads.@spawn begin
            id = Threads.threadid()
            _mul_recursive!(Cthreads[id],block,B)
        end
    end
    # reduce
    for Ct in Cthreads
        axpy!(1,Ct,C)
    end
    return C
end

# multiply in parallel using a static partitioning of the leaves computed "by
# hand" in partition
function _mul_static!(C::AbstractVector,A::HMatrix,B::AbstractVector,partition)
    # multiply by b at root level
    # rmul!(C,b)
    # create a lock for the reduction step
    mutex = ReentrantLock()
    nt    = length(partition)
    times = Vector{Float64}(undef,nt)
    Threads.@threads for n in 1:nt
        id = Threads.threadid()
        times[id] =
        @elapsed begin
            leaves = partition[n]
            Cloc   = zero(C)
            for leaf in leaves
                irange = rowrange(leaf)
                jrange = colrange(leaf)
                data   = leaf.data
                mul!(view(Cloc,irange),data,view(B,jrange),1,1)
            end
            # reduction
            lock(mutex) do
                axpy!(1,Cloc,C)
            end
        end
        # @debug "Matrix vector product" Threads.threadid() times[id]
    end
    tmin,tmax = extrema(times)
    if tmax/tmin > 1.1
        @warn "gemv: ratio of tmax/tmin = $(tmax/tmin)"
    end
    # @debug "Gemv: tmin = $tmin, tmax = $tmax, ratio = $((tmax)/(tmin))"
    return C
end

function LinearAlgebra.mul!(C::AbstractVector,A::HMatrix,B::AbstractVector,a::Number,b::Number)
    # since the HMatrix represents A = Pr*H*Pc, where Pr and Pc are row and column
    # permutations, we need first to rewrite C <-- b*C + a*(Pc*H*Pb)*B as
    # C <-- Pr*(b*inv(Pr)*C + a*H*(Pc*B)). Following this rewrite, the
    # multiplication is performed by first defining B <-- Pc*B, and C <--
    # inv(Pr)*C, doing the multiplication with the permuted entries, and then
    # permuting the result  C <-- Pr*C at the end. This is controlled by the
    # flat `P`
    ctree     = A.coltree
    rtree     = A.rowtree
    # permute input
    B         = B[ctree.loc2glob]
    C         = permute!(C,rtree.loc2glob)
    rmul!(B,a)
    rmul!(C,b)
    # _mul_CPU!(C,A,B,1,1)
    # _mul_threads!(C,A,B,a,b)
    nt        = Threads.nthreads()
    partition = hilbert_partitioning(A,nt)
    _mul_static!(C,A,B,partition)
    # permute output
    permute!(C,rtree.glob2loc)
end

function Base.:(*)(H::HMatrix,x::AbstractVector)
    T  = eltype(H)
    S  = eltype(x)
    TS = promote_type(T,S)
    y  = Vector{TS}(undef,size(H,1))
    mul!(y,H,x)
end

"""
    hilbert_partitioning(H::HMatrix,np,[cost=cost_mv])

Partiotion the leaves of `H` into `np` sequences of approximate equal cost (as
determined by the `cost` function) while also trying to maximize the locality of
each partition.
"""
function hilbert_partitioning(H::HMatrix,np=Threads.nthreads(),cost=cost_mv)
    # the hilbert curve will be indexed from (0,0) × (N-1,N-1), so set N to be
    # the smallest power of two larger than max(m,n), where m,n = size(H)
    m,n = size(H)
    N   = max(m,n)
    N   = nextpow(2,N)
    # sort the leaves by their hilbert index
    leaves = getnodes(x -> isleaf(x),H)
    hilbert_indices = map(leaves) do leaf
        # use the center of the leaf as a cartesian index
        i,j = pivot(leaf) .- 1 .+ size(leaf) .÷ 2
        hilbert_cartesian_to_linear(N,i,j)
    end
    p = sortperm(hilbert_indices)
    permute!(leaves,p)
    # now compute a quasi-optimal partition of leaves based `cost_mv`
    cmax      = find_optimal_cost(leaves,np,cost,1)
    partition = build_sequence_partition(leaves,np,cost,cmax)
    return partition
end

"""
    build_sequence_partition(seq,nq,cost,nmax)

Partition the sequence `seq` into `nq` contiguous subsequences with a maximum of
cost of `nmax` per set. Note that if `nmax` is too small, this may not be
possible (see [`has_partition`](@ref)).
"""
function build_sequence_partition(seq,np,cost,cmax)
    acc = 0
    partition = [empty(seq) for _ in 1:np]
    k = 1
    for el in seq
        c = cost(el)
        acc += c
        if acc > cmax
            k += 1
            push!(partition[k],el)
            acc = c
            @assert k <= np "unable to build sequence partition. Value of  `cmax` may be too small."
        else
            push!(partition[k],el)
        end
    end
    return partition
end

"""
    find_optimal_cost(seq,nq,cost,tol)

Find an approximation to the cost of an optimal partitioning of `seq` into `nq`
contiguous subsequent. The optimal cost is the smallest number `cmax` such that
`has_partition(seq,nq,cost,cmax)` returns `true`.
"""
function find_optimal_cost(seq,np,cost=identity,tol=1)
    lbound = maximum(cost,seq) |> Float64
    ubound = sum(cost,seq)     |> Float64
    guess  = (lbound+ubound)/2
    while ubound - lbound ≥ tol
        if has_partition(seq,np,guess,cost)
            ubound = guess
        else
            lbound = guess
        end
        guess  = (lbound+ubound)/2
    end
    return ubound
end


"""
    has_partition(v,np,cost,cmax)

Given a vector `v`, determine whether or not a partition into `np` segments is
possible where the `cost` of each partition does not exceed `cmax`.
"""
function has_partition(seq,np,cmax,cost=identity)
    acc = 0
    k   = 1
    for el in seq
        c = cost(el)
        acc += c
        if acc > cmax
            k += 1
            acc = c
            k > np && (return false)
        end
    end
    return true
end


function row_partitioning(H::HMatrix,np=Threads.nthreads())
    # sort the leaves by their row index
    leaves = getnodes(x -> isleaf(x),H)
    row_indices = map(leaves) do leaf
        # use the center of the leaf as a cartesian index
        i,j = pivot(leaf)
        return i
    end
    p = sortperm(row_indices)
    permute!(leaves,p)
    # now compute a quasi-optimal partition of leaves based `cost_mv`
    cmax = find_optimal_cost(leaves,np,cost_mv,1)
    partition = build_sequence_partition(leaves,np,cost_mv,cmax)
    return partition
end

function col_partitioning(H::HMatrix,np=Threads.nthreads())
    # sort the leaves by their row index
    leaves = getnodes(x -> isleaf(x),H)
    row_indices = map(leaves) do leaf
        # use the center of the leaf as a cartesian index
        i,j = pivot(leaf)
        return j
    end
    p = sortperm(row_indices)
    permute!(leaves,p)
    # now compute a quasi-optimal partition of leaves based `cost_mv`
    cmax = find_optimal_cost(leaves,np,cost_mv,1)
    partition = build_sequence_partition(leaves,np,cost_mv,cmax)
    return partition
end

"""
    cost_mv(A::AbstractMatrix)

A proxy for the computational cost of a matrix/vector product.
"""
function cost_mv(R::RkMatrix)
    rank(R)*sum(size(R))
end
function cost_mv(M::Base.Matrix)
    length(M)
end
function cost_mv(H::HMatrix)
    cost_mv(H.data)
end
