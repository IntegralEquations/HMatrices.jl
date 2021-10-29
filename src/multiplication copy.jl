############################################################################################
####################################### rmul! ##############################################
############################################################################################
function LinearAlgebra.rmul!(R::RkMatrix,b::Number)
    m,n = size(R)
    if m>n
        rmul!(R.B,conj(b))
    else
        rmul!(R.A,b)
    end
    return R
end

function LinearAlgebra.rmul!(H::HMatrix,b::Number)
    b==true && (return H) # short circuit. If inlined, rmul!(H,true) --> no-op
    if hasdata(H)
        rmul!(data(H),b)
    end
    for child in children(H)
        rmul!(child,b)
    end
    return H
end

# ############################################################################################
# ####################################### Base.:* ############################################
# ############################################################################################

Base.:(*)(a::Number,R::RkMatrix) = rmul!(deepcopy(R),a)
Base.:(*)(R::RkMatrix,a::Number) = rmul!(deepcopy(R),a)

"""
    (*)(A::Union{Matrix,RkMatrix,HMatrix},B::Union{Matrix,RkMatrix,HMatrix}) --> C

Two argument multiplication. When operating on `Union{Matrix,RkMatrix,HMatrix}`,
the result `C` is returned in the *natural format*, as described in the table
below:

| `(*)(A,B)` | `B::Matrix`  | `B::RkMatrix`   | `B::HMatrix` |
|:-----|:---:|:---:|:---:|
|`A::Matrix`  | `C::Matrix`  | `C::RkMatrix` | `C::Matrix` |
|`A::RkMatrix`  | `C::RkMatrix` | `C::RkMatrix` | `C::RkMatrix` |
|`A::HMatrix`  | `C::Matrix` | `C::RkMatrix` | `C::HMatrix` |
"""
function Base.:(*)(M::Base.Matrix,R::RkMatrix)
    tmp = M*R.A
    return RkMatrix(tmp,copy(R.B))
end

function Base.:(*)(R1::RkMatrix,R2::RkMatrix)
    if rank(R1) < rank(R2)
        RkMatrix(R1.A, R2.B*(R2.At*R1.B))
    else
        RkMatrix(R1.A*(R1.Bt*R2.A) , R2.B)
    end
end

# ############################################################################################
# ####################################### mul! ###############################################
# ############################################################################################

##################################################################################
#                   mul!(C,A,B,a,b) :  C <-- b*C + a*A*B
# For the mul!(C,A,B,a,b) function, there are 3^3 = 27 cases to be considered
# depending on the types of C, A, and B. We will list these cases by by x.x.x
# where 1 means a full matrix, 2 a sparse matrix, and 3 a hierarhical matrix.
# E.g. case 1.2.1 means C is full, A is sparse, B is full.
# Note: no re-compression is performed after mul!
##################################################################################

# """
#     struct HMulTree{T} <: AbstractTree

# Structure used to group the operations which need to be performed during
# `mul!(C::HMatrix,A::HMatrix,B::HMatrix,a,true)`. After creation, call
# `evaluate(::HMulTree,a,tol)` to perform the multiplication.
# """
# struct HMulTree{T} <: AbstractTree
#     node::T
#     children::Matrix{T}
#     pairs::Vector{Tuple{T,T}}
# end

# function evaluate!(mtree::HMulTree,a,tol)
#     Cdata = hasdata(C.node) ? data(mtree.node) : C
#     if Cdata isa Matrix
#         _mul_matrix!(C,mtree.pairs,a,true)
#     elseif C isa RkMatrix
#         _mul_rkmatrix!(C,mtree.pairs,a,true)
#     elseif C isa HMatrix
#     else
#         error()
#     end
# end

# # target is a full matrix
# function _mul_matrix!(C,pairs,a,b)
#     T = eltype(C)
#     for (A,B) in pairs
#         # A is a full matrix
#         if A isa Matrix
#             if B isa Matrix
#                 mul!(C,A,B,a,b)
#             elseif B isa RkMatrix
#                 mul!(C,A*B.A,B.Bt,a,b)
#             elseif B isa HMatrix
#                 tmp = A*B
#                 axpby!(a,tmp,b,C)
#             end
#         elseif A isa RkMatrix
#             if B isa Matrix
#                 mul!(C,A.A,A.Bt*B,a,b)
#             elseif B isa RkMatrix
#                 tmp = (A.Bt*B.A)*B.Bt
#                 mul!(C,A.A,tmp,a,b)
#             elseif B isa HMatrix
#                 tmp = A.Bt*B
#                 mul!(C,A.A,tmp,a,b)
#             end
#         elseif A isa HMatrix
#             if B isa Matrix
#                 tmp = A*B
#                 axpby!(a,tmp,b,C)
#             elseif B isa RkMatrix
#                 tmp = A*B.A
#                 mul!(C,tmp,B.Bt,a,b)
#             elseif B isa HMatrix
#                 error("this case should not happen")
#             end
#         end
#     end
#     return C
# end

# # target is an rk-matrix
# function _mul_rkmatrix!(C,pairs,a,b,comp)
#     m,n = size(C)
#     T = eltype(C)
#     acc_full = (undef,m,n)
#     for (A,B) in pairs
#         # A is a full matrix
#         if A isa Matrix
#             if B isa Matrix
#                 mul!(acc_full,A,B)
#             elseif B isa RkMatrix
#                 mul!(C,A*B.A,B.Bt,a,b)
#             elseif B isa HMatrix
#                 tmp = A*B
#                 axpby!(a,tmp,b,C)
#             end
#         elseif A isa RkMatrix
#             if B isa Matrix
#                 mul!(C,A.A,A.Bt*B,a,b)
#             elseif B isa RkMatrix
#                 tmp = (A.Bt*B.A)*B.Bt
#                 mul!(C,A.A,tmp,a,b)
#             elseif B isa HMatrix
#                 tmp = A.Bt*B
#                 mul!(C,A.A,tmp,a,b)
#             end
#         elseif A isa HMatrix
#             if B isa Matrix
#                 tmp = A*B
#                 axpby!(a,tmp,b,C)
#             elseif B isa RkMatrix
#                 tmp = A*B.A
#                 mul!(C,tmp,B.Bt,a,b)
#             elseif B isa HMatrix
#                 error("this case should not happen")
#             end
#         end
#     end
#     return C
# end


# ################################################################################
# ## 1.1.1
# ################################################################################
# # default multiplication of dense matrices

################################################################################
## 1.1.2
################################################################################
function LinearAlgebra.mul!(C::AbstractMatrix,M::AbstractMatrix,R::RkMatrix,a::Number,b::Number)
    buffer = M*R.A
    mul!(C,buffer,R.Bt,a,b)
    return C
end

################################################################################
## 1.1.3
################################################################################x
function LinearAlgebra.mul!(C::AbstractMatrix,M::AbstractMatrix,H::HMatrix,a::Number,b::Number)
    rmul!(C,b)
    if hasdata(H)
        mat = data(H)
        mul!(C,M,mat,a,true)
    end
    for child in children(H)
        shift  = pivot(H) .- 1
        irange = rowrange(child) .- shift[1]
        jrange = colrange(child) .- shift[2]
        Cview  = view(C,:,jrange)
        Mview  = view(M,:,irange)
        mul!(Cview,Mview,child,a,true)
    end
    return C
end

################################################################################
## 1.2.1
################################################################################
function LinearAlgebra.mul!(C::AbstractMatrix,R::RkMatrix,M::AbstractMatrix,a::Number,b::Number)
    buffer = R.Bt*M
    mul!(C,R.A,buffer,a,b)
end
function LinearAlgebra.mul!(C::AbstractMatrix,adjR::Adjoint{<:Any,<:RkMatrix},M::AbstractMatrix,a::Number,b::Number)
    R   = LinearAlgebra.parent(adjR)
    tmp = adjoint(R.A)*M
    mul!(C,R.B,tmp,a,b)
    return C
end

################################################################################
## 1.2.2
################################################################################
function LinearAlgebra.mul!(C::AbstractMatrix,R::RkMatrix,S::RkMatrix,a::Number,b::Number)
    tmp = R*S
    mul!(C,tmp.A,tmp.Bt,a,b)
    return C
end

################################################################################
## 1.2.3
################################################################################
function LinearAlgebra.mul!(C::AbstractMatrix,R::RkMatrix,H::HMatrix,a::Number,b::Number)
    T = promote_type(eltype(R),eltype(H))
    tmp = zeros(T,size(R.Bt,1),size(H,2))
    mul!(tmp,R.Bt,H)
    mul!(C,R.A,tmp,a,b)
    return C
end

################################################################################
## 1.3.1
################################################################################
function LinearAlgebra.mul!(C::AbstractMatrix,H::HMatrix,M::AbstractMatrix,a::Number,b::Number)
    rmul!(C,b)
    if hasdata(H)
        mat = data(H)
        mul!(C,mat,M,a,true)
    end
    for child in children(H)
        shift  = pivot(H) .- 1
        irange = rowrange(child) .- shift[1]
        jrange = colrange(child) .- shift[2]
        Cview  = view(C,irange,:)
        Mview  = view(M,jrange,:)
        mul!(Cview,child,Mview,a,true)
    end
    return C
end

# function LinearAlgebra.mul!(C::AbstractMatrix,adjH::Adjoint{<:Any,<:HMatrix},M::AbstractMatrix,a::Number,b::Number)
#     rmul!(C,b)
#     if hasdata(adjH)
#         data = getdata(adjH)
#         mul!(C,data,M,a,true)
#     end
#     for child in children(adjH)
#         shift  = pivot(adjH) .- 1
#         irange = rowrange(child) .- shift[1]
#         jrange = colrange(child) .- shift[2]
#         Cview  = view(C,irange,:)
#         Mview  = view(M,jrange,:)
#         mul!(Cview,child,Mview,a,true)
#     end
#     return C
# end

################################################################################
## 1.3.2
################################################################################
function LinearAlgebra.mul!(C::AbstractMatrix,H::HMatrix,R::RkMatrix,a::Number,b::Number)
    buffer=similar(C,size(H,1),size(R.A,2))
    mul!(buffer,H,R.A)
    mul!(C,buffer,R.Bt,a,b)
    return C
end

################################################################################
## 1.3.3 (should never arise in practice, thus sloppy implementation)
################################################################################
function LinearAlgebra.mul!(C::AbstractMatrix,H::HMatrix,S::HMatrix,a::Number,b::Number)
    error("1.3.3: this case should not arise")
    # mul!(C,H,Matrix(S),a,b)
    # return C
end

################################################################################
## 2.1.1
################################################################################
function LinearAlgebra.mul!(C::RkMatrix,M::AbstractMatrix,F::AbstractMatrix,a::Number,b::Number)
    error("should not happen")
    # buffer=similar(C,size(M,1),size(F,2))
    # mul!(buffer,M,F)
    # axpby!(a,buffer,b,C)
    # return C
end

################################################################################
## 2.1.2
################################################################################
function LinearAlgebra.mul!(C::RkMatrix,M::AbstractMatrix,R::RkMatrix,a::Number,b::Number)
    tmp = RkMatrix(M*R.A,R.B)
    axpby!(a,tmp,b,C)
end

################################################################################
## 2.1.3
################################################################################
function LinearAlgebra.mul!(C::RkMatrix,M::AbstractMatrix,H::HMatrix,a::Number,b::Number)
    error("2.1.3: this case should not arise")
    # buffer = similar(C,size(M,1),size(H,2))
    # mul!(buffer,M,H)
    # axpby!(a,buffer,b,C)
    # return C
end

################################################################################
## 2.2.1
################################################################################
function LinearAlgebra.mul!(C::RkMatrix,R::RkMatrix,M::AbstractMatrix,a::Number,b::Number)
    tmp = RkMatrix(R.A,adjoint(M)*R.B)
    axpby!(a,tmp,b,C)
    return C
end

################################################################################
## 2.2.2
################################################################################
function LinearAlgebra.mul!(C::RkMatrix,R::RkMatrix,S::RkMatrix,a::Number,b::Number)
    tmp = R*S
    axpby!(a,tmp,b,C)
    return C
end

################################################################################
## 2.2.3
################################################################################
function LinearAlgebra.mul!(C::RkMatrix,R::RkMatrix,H::HMatrix,a::Number,b::Number)
    T      = promote_type(eltype(R),eltype(H))
    buffer = zeros(T,size(R.Bt,1),size(H,2))
    mul!(buffer,R.Bt,H)
    tmp = RkMatrix(R.A,collect(adjoint(buffer)))
    axpby!(a,tmp,b,C)
    return C
end

################################################################################
## 2.3.1
################################################################################
function LinearAlgebra.mul!(C::RkMatrix,H::HMatrix,M::AbstractMatrix,a::Number,b::Number)
    error("2.3.1: this case should not arise")
    # T = promote_type(eltype(H),eltype(M))
    # buffer = Matrix{T}(undef,size(H,1),size(M,2))
    # mul!(buffer,H,M)
    # axpby!(a,buffer,b,C)
    # return C
end

################################################################################
## 2.3.2
################################################################################
function LinearAlgebra.mul!(C::RkMatrix,H::HMatrix,R::RkMatrix,a::Number,b::Number)
    T = promote_type(eltype(H),eltype(R))
    buffer = zeros(T,size(H,1),size(R.A,2))
    mul!(buffer,H,R.A)
    tmp = RkMatrix(buffer,R.B)
    axpby!(a,tmp,b,C)
    return C
end

################################################################################
## 2.3.3
################################################################################
function LinearAlgebra.mul!(C::RkMatrix,A::HMatrix,B::HMatrix,a::Number,b::Number)
    rmul!(C,b)
    if !isleaf(A) && !isleaf(B)
        m,n    = blocksize(A,1), blocksize(B,2)
        block  = Matrix{typeof(C)}(undef,m,n)
        for i = 1:m
            for j = 1:n
                p = size(getblock(A,i,1),1)
                q = size(getblock(B,1,j),2)
                block[i,j] = zero(typeof(C),p,q)
                for k = 1:blocksize(A,2)
                    mul!(block[i,j],getblock(A,i,k),getblock(B,k,j),true,true)
                end
            end
        end
        R = _gather(block)
        axpby!(a,R,true,C)
        # C.A = R.A
        # C.B = R.B
    else
        Adata = isleaf(A) ? A.data : A
        Bdata = isleaf(B) ? B.data : B
        mul!(C,Adata,Bdata,a,true)
    end
    return C
end

# ################################################################################
# ## 3.1.1
# ################################################################################
# function LinearAlgebra.mul!(C::HMatrix,M::AbstractMatrix,F::AbstractMatrix,a::Number,b::Number)
#     tmp = M*F
#     if hasdata(C)
#         axpby!(a,tmp,b,C.data)
#     else
#         rmul!(C,b)
#         C.data = rmul!(tmp,a)
#     end
#     return C
# end

# ################################################################################
# ## 3.1.2
# ################################################################################
# function LinearAlgebra.mul!(C::HMatrix,M::AbstractMatrix,R::RkMatrix,a::Number,b::Number)
#     tmp = M*R
#     if hasdata(C)
#         axpby!(a,tmp,b,C.data)
#     else
#         rmul!(C,b)
#         C.data = rmul!(tmp,a)
#     end
#     return C
# end

# ################################################################################
# ## 3.1.3
# ################################################################################
# function LinearAlgebra.mul!(C::HMatrix,M::AbstractMatrix,H::HMatrix,a::Number,b::Number)
#     tmp = M*H
#     if hasdata(C)
#         axpby!(a,tmp,b,C.data)
#     else
#         rmul!(C,b)
#         C.data = rmul!(tmp,a)
#     end
#     return C
# end

# ################################################################################
# ## 3.2.1
# ################################################################################
# function LinearAlgebra.mul!(C::HMatrix,R::RkMatrix,M::AbstractMatrix,a::Number,b::Number)
#     tmp = R*M
#     if hasdata(C)
#         axpby!(a,tmp,b,C.data)
#     else
#         rmul!(C,b)
#         C.data = rmul!(tmp,a)
#     end
#     return C
# end

# ################################################################################
# ## 3.2.2
# ################################################################################
# function LinearAlgebra.mul!(C::HMatrix,R::RkMatrix,M::RkMatrix,a::Number,b::Number)
#     tmp = R*M
#     if hasdata(C)
#         axpby!(a,tmp,b,C.data)
#     else
#         rmul!(C,b)
#         C.data = rmul!(tmp,a)
#     end
#     return C
# end

# ################################################################################
# ## 3.2.3
# ################################################################################
# function LinearAlgebra.mul!(C::HMatrix,R::RkMatrix,H::HMatrix,a::Number,b::Number)
#     tmp = R*H
#     if hasdata(C)
#         axpby!(a,tmp,b,C.data)
#     else
#         rmul!(C,b)
#         C.data = rmul!(tmp,a)
#     end
#     return C
# end

# ################################################################################
# ## 3.3.1
# ################################################################################
# function LinearAlgebra.mul!(C::HMatrix,H::HMatrix,M::Matrix,a::Number,b::Number)
#     @debug "3.3.1: this case should not arise"
#     tmp = H*M
#     if hasdata(C)
#         axpby!(a,tmp,b,C.data)
#     else
#         rmul!(C,b)
#         C.data = rmul!(tmp,a)
#     end
#     return C
# end

# ################################################################################
# ## 3.3.2
# ################################################################################
# function LinearAlgebra.mul!(C::HMatrix,H::HMatrix,R::RkMatrix,a::Number,b::Number)
#     tmp = H*R
#     if hasdata(C)
#         axpby!(a,tmp,b,C.data)
#     else
#         rmul!(C,b)
#         C.data = rmul!(tmp,a)
#     end
#     return C
# end

# ################################################################################
# ## 3.3.3 (the recursive case)
# ################################################################################
# function LinearAlgebra.mul!(C::HMatrix,A::HMatrix,B::HMatrix,a::Number,b::Number,compress=identity)
#     rmul!(C,b)
#     # when either A, B, or C is a leaf, extract the data and dispatch to one of the other
#     # mul! methods above. Otherwise recurse on children.
#     if isleaf(A) || isleaf(B) || isleaf(B)
#         _mul_leaf!(C,A,B,a,true,compress)
#     else
#         ni,nj = blocksize(C)
#         _ ,nk = blocksize(A)
#         A_children = children(A)
#         B_children = children(B)
#         C_children = children(C)
#         for i=1:ni
#             for j=1:nj
#                 for k=1:nk
#                     mul!(C_children[i,j],A_children[i,k],B_children[k,j],a,true,tol,compress)
#                 end
#             end
#         end
#     end
#     return C
# end

# # terminal case which dynamically dispatches to appropriate method
# function _mul_leaf!(C::HMatrix,A::HMatrix,B::HMatrix,a::Number,b::Number,compress)
#     @assert isleaf(A) || isleaf(B) || isleaf(B)
#     Cdata  = hasdata(C) ? data(C) : C
#     Adata  = hasdata(A) ? data(A) : A
#     Bdata  = hasdata(B) ? data(B) : B
#     mul!(Cdata,Adata,Bdata,a,true,compress)
#     flush_tree!(C)
#     return C
# end

# ############################################################################################
# # Specializations on gemv
# ############################################################################################

# # R*x
# function LinearAlgebra.mul!(C::AbstractVector,Rk::RkMatrix,F::AbstractVector,a::Number,b::Number)
#     tmp = Rk.Bt*F
#     mul!(C,Rk.A,tmp,a,b)
# end

# # Rt*x
# function LinearAlgebra.mul!(y::AbstractVector,Rt::Adjoint{<:Any,<:RkMatrix},x::AbstractVector,a::Number,b::Number)
#     R  = Rt.parent
#     At = adjoint(R.A)
#     buffer = At*x
#     mul!(y,R.B,buffer,a,b)
#     return y
# end

# # xt*R
# const VecAdj{T} = Adjoint{T,Vector{T}}
# function LinearAlgebra.mul!(yt::VecAdj,xt::VecAdj,R::RkMatrix,a::Number,b::Number)
#     mul!(yt.parent,adjoint(R),xt.parent,a,b)
#     return yt
# end

# # H*x
# function LinearAlgebra.mul!(C::AbstractVector,A::HMatrix,B::AbstractVector,a::Number,b::Number)
#     # since the HMatrix represents A = Pr*H*Pc, where Pr and Pc are row and column
#     # permutations, we need first to rewrite C <-- b*C + a*(Pc*H*Pb)*B as
#     # C <-- Pr*(b*inv(Pr)*C + a*H*(Pc*B)). Following this rewrite, the
#     # multiplication is performed by first defining B <-- Pc*B, and C <--
#     # inv(Pr)*C, doing the multiplication with the permuted entries, and then
#     # permuting the result  C <-- Pr*C at the end. This is controlled by the
#     # flat `P`
#     ctree     = A.coltree
#     rtree     = A.rowtree
#     # permute input
#     B         = B[ctree.loc2glob]
#     C         = permute!(C,rtree.loc2glob)
#     rmul!(B,a)
#     rmul!(C,b)
#     # _mul_recursive(C,A,B) # serial implementation
#     # _mul_threads!(C,A,B)  # threaded implementation
#     # nt        = Threads.nthreads()
#     # partition = hilbert_partitioning(A,nt)
#     # _mul_static!(C,A,B,partition) # threaded implementation, by hand partition
#     # permute output
#     invpermute!(C,loc2glob(rtree))
# end


# ############################################################################################
# ############################# auxiliary functions ##########################################
# ############################################################################################

# function _mul_threads!(C::AbstractVector,A::HMatrix,B::AbstractVector)
#     # make copies of C and run in parallel
#     nt        = Threads.nthreads()
#     Cthreads  = [zero(C) for _ in 1:nt]
#     blocks    = filter(x -> isleaf(x),A)
#     @sync for block in blocks
#         Threads.@spawn begin
#             id = Threads.threadid()
#             _mul_recursive!(Cthreads[id],block,B)
#         end
#     end
#     # reduce
#     for Ct in Cthreads
#         axpy!(1,Ct,C)
#     end
#     return C
# end

# # multiply in parallel using a static partitioning of the leaves computed "by
# # hand" in partition
# function _mul_static!(C::AbstractVector,A::HMatrix,B::AbstractVector,partition)
#     # multiply by b at root level
#     # rmul!(C,b)
#     # create a lock for the reduction step
#     mutex = ReentrantLock()
#     nt    = length(partition)
#     times = Vector{Float64}(undef,nt)
#     Threads.@threads for n in 1:nt
#         id = Threads.threadid()
#         times[id] =
#         @elapsed begin
#             leaves = partition[n]
#             Cloc   = zero(C)
#             for leaf in leaves
#                 irange = rowrange(leaf)
#                 jrange = colrange(leaf)
#                 data   = leaf.data
#                 mul!(view(Cloc,irange),data,view(B,jrange),1,1)
#             end
#             # reduction
#             lock(mutex) do
#                 axpy!(1,Cloc,C)
#             end
#         end
#         # @debug "Matrix vector product" Threads.threadid() times[id]
#     end
#     tmin,tmax = extrema(times)
#     if tmax/tmin > 1.1
#         @warn "gemv: ratio of tmax/tmin = $(tmax/tmin)"
#     end
#     # @debug "Gemv: tmin = $tmin, tmax = $tmax, ratio = $((tmax)/(tmin))"
#     return C
# end

# function _mul_recursive!(C::AbstractVector,A::HMatrix,B::AbstractVector)
#     T = eltype(A)
#     if isleaf(A)
#         irange = rowrange(A)
#         jrange = colrange(A)
#         data   = A.data
#         if T <: Number
#             LinearAlgebra.mul!(view(C,irange),data,view(B,jrange),1,1)
#         elseif T<: SMatrix
#             # see this issue:
#             if data isa Matrix
#                 LinearAlgebra.mul!(view(C,irange,1:1),data,view(B,jrange,1:1),1,1)
#             else
#                 LinearAlgebra.mul!(view(C,irange),data,view(B,jrange),1,1)
#             end
#         else
#             error("T=$T")
#         end
#     else
#         for block in A.children
#             _mul_recursive!(C,block,B)
#         end
#     end
#     return C
# end

# """
#     hilbert_partitioning(H::HMatrix,np,[cost=cost_mv])

# Partiotion the leaves of `H` into `np` sequences of approximate equal cost (as
# determined by the `cost` function) while also trying to maximize the locality of
# each partition.
# """
# function hilbert_partitioning(H::HMatrix,np=Threads.nthreads(),cost=cost_mv)
#     # the hilbert curve will be indexed from (0,0) × (N-1,N-1), so set N to be
#     # the smallest power of two larger than max(m,n), where m,n = size(H)
#     m,n = size(H)
#     N   = max(m,n)
#     N   = nextpow(2,N)
#     # sort the leaves by their hilbert index
#     leaves = filter(x -> isleaf(x),H)
#     hilbert_indices = map(leaves) do leaf
#         # use the center of the leaf as a cartesian index
#         i,j = pivot(leaf) .- 1 .+ size(leaf) .÷ 2
#         hilbert_cartesian_to_linear(N,i,j)
#     end
#     p = sortperm(hilbert_indices)
#     permute!(leaves,p)
#     # now compute a quasi-optimal partition of leaves based `cost_mv`
#     cmax      = find_optimal_cost(leaves,np,cost,1)
#     partition = build_sequence_partition(leaves,np,cost,cmax)
#     return partition
# end

# function row_partitioning(H::HMatrix,np=Threads.nthreads())
#     # sort the leaves by their row index
#     leaves = filter(x -> isleaf(x),H)
#     row_indices = map(leaves) do leaf
#         # use the center of the leaf as a cartesian index
#         i,j = pivot(leaf)
#         return i
#     end
#     p = sortperm(row_indices)
#     permute!(leaves,p)
#     # now compute a quasi-optimal partition of leaves based `cost_mv`
#     cmax = find_optimal_cost(leaves,np,cost_mv,1)
#     partition = build_sequence_partition(leaves,np,cost_mv,cmax)
#     return partition
# end

# function col_partitioning(H::HMatrix,np=Threads.nthreads())
#     # sort the leaves by their row index
#     leaves = filter(x -> isleaf(x),H)
#     row_indices = map(leaves) do leaf
#         # use the center of the leaf as a cartesian index
#         i,j = pivot(leaf)
#         return j
#     end
#     p = sortperm(row_indices)
#     permute!(leaves,p)
#     # now compute a quasi-optimal partition of leaves based `cost_mv`
#     cmax = find_optimal_cost(leaves,np,cost_mv,1)
#     partition = build_sequence_partition(leaves,np,cost_mv,cmax)
#     return partition
# end

# """
#     cost_mv(A::AbstractMatrix)

# A proxy for the computational cost of a matrix/vector product.
# """
# function cost_mv(R::RkMatrix)
#     rank(R)*sum(size(R))
# end
# function cost_mv(M::Base.Matrix)
#     length(M)
# end
# function cost_mv(H::HMatrix)
#     cost_mv(H.data)
# end

# ################################################################################
# ## FLUSH_TO_CHILDREN
# ################################################################################
# function flush_to_children!(H::HMatrix)
#     hasdata(H) && !isleaf(H) || (return H)
#     R     = getdata(H)
#     add_to_children!(H,R)
#     setdata!(H,())
#     return H
# end

# function add_to_children!(H,R::RkMatrix)
#     shift = pivot(H) .- 1
#     for block in getchildren(H)
#         irange     = rowrange(block) .- shift[1]
#         jrange     = colrange(block) .- shift[2]
#         bdata      = getdata(block)
#         tmp        = RkMatrix(R.A[irange,:],R.B[jrange,:])
#         if bdata === ()
#             setdata!(block,tmp)
#         else
#             axpby!(true,tmp,true,bdata)
#         end
#     end
# end

# function add_to_children!(H,M::Matrix)
#     shift = pivot(H) .- 1
#     for block in getchildren(H)
#         irange     = rowrange(block) .- shift[1]
#         jrange     = colrange(block) .- shift[2]
#         bdata      = getdata(block)
#         tmp        = M[irange,jrange]
#         if bdata === ()
#             setdata!(block,tmp)
#         else
#             axpby!(true,tmp,true,bdata)
#         end
#     end
# end

# ################################################################################
# ## FLUSH_TREE
# ################################################################################
# function flush_tree!(H::HMatrix)
#     flush_to_children!(H)
#     for block in getchildren(H)
#         flush_tree!(block)
#     end
#     return H
# end

# ################################################################################
# ## FLUSH_TO_LEAVES
# ################################################################################
# function flush_to_leaves!(H::HMatrix)
#     hasdata(H) && !isleaf(H) || (return H)
#     R = getdata(H)
#     add_to_leaves!(H,R)
#     H.data = ()
#     return H
# end

# function add_to_leaves!(H::HMatrix,R::RkMatrix)
#     shift = pivot(H) .- 1
#     for block in Leaves(H)
#         irange     = rowrange(block) .- shift[1]
#         jrange     = colrange(block) .- shift[2]
#         bdata      = getdata(block)
#         tmp        = RkMatrix(R.A[irange,:],R.B[jrange,:])
#         if bdata === ()
#             setdata!(block,tmp)
#         else
#             axpby!(true,tmp,true,bdata)
#         end
#     end
#     return H
# end

# function add_to_leaves!(H::HMatrix,M::Matrix)
#     shift = pivot(H) .- 1
#     for block in Leaves(H)
#         irange     = rowrange(block) .- shift[1]
#         jrange     = colrange(block) .- shift[2]
#         bdata      = getdata(block)
#         tmp        = M[irange,jrange]
#         if bdata === ()
#             setdata!(block,tmp)
#         else
#             axpby!(true,tmp,true,bdata)
#         end
#     end
#     return H
# end
