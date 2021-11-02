"""
    hmul!(C::HMatrix,A::HMatrix,B::HMatrix,a,b,compressor)

Similar to `LinearAlgebra.mul!` : compute `C <-- A*B*a + B*b`, where `A,B,C` are
hierarchical matrices and `compressor` is a function/functor used in the
intermediate stages of the multiplication to avoid growring the rank of
admissible blocks after addition is performed.
"""
function hmul!(C::HMatrix,A::HMatrix,B::HMatrix,a,b,compressor)
    b == true || rmul!(C,b)
    plan = plan_hmul(C,A,B,a,1)
    execute!(plan,compressor)
    # deprecated version of mul!
    # _mul333!(C,A,B,a,compressor)
    return C
end
# disable `mul` of hierarchial matrices
function LinearAlgebra.mul!(C::HMatrix,A::HMatrix,B::HMatrix,a::Number,b::Number)
    msg = "use `hmul` to multiply hierarchical matrices"
    error(msg)
end

##################################################################################
# The methods below implement what is required to perform the mul! of
# hierarchical matrices. There are 3^3 = 27 cases to be considered
# depending on the types of C, A, and B. We will list these cases by by x.x.x
# where 1 means a full matrix, 2 a sparse matrix, and 3 a hierarhical matrix.
# E.g. case 1.2.1 means C is full, A is sparse, B is full.
# Note: initially these were all methods of the `mul!` function, but it turned
# out to be a mess to debug and profile 27 methods with the same name, so I
# decided to name them differently. One downside is that some of the tasks that
# could be automatically handled by the dynamic dispatch system have to be done
# manually...
##################################################################################

# ################################################################################
# ## 1.1.1
# ################################################################################
_mul111!(C::Union{Matrix,SubArray,Adjoint},A::Union{Matrix,SubArray,Adjoint},B::Union{Matrix,SubArray,Adjoint},a::Number) = mul!(C,A,B,a,true)

################################################################################
## 1.1.2
################################################################################
function _mul112!(C::Union{Matrix,SubArray,Adjoint}, M::Union{Matrix,SubArray,Adjoint}, R::RkMatrix, a::Number)
    buffer = M * R.A
    _mul111!(C, buffer, R.Bt, a)
    return C
end

################################################################################
## 1.1.3
################################################################################x
function _mul113!(C::Union{Matrix,SubArray,Adjoint}, M::Union{Matrix,SubArray,Adjoint}, H::HMatrix, a::Number)
    T = eltype(C)
    if hasdata(H)
        mat = data(H)
        if mat isa Matrix
            _mul111!(C, M, mat, a)
        elseif mat isa RkMatrix
            _mul112!(C, M, mat, a)
        else
            error()
        end
    end
    for child in children(H)
        shift  = pivot(H) .- 1
        irange = rowrange(child) .- shift[1]
        jrange = colrange(child) .- shift[2]
        Cview  = @views C[:, jrange]
        Mview  = @views M[:, irange]
        _mul113!(Cview, Mview, child, a)
    end
    return C
end

################################################################################
## 1.2.1
################################################################################
function _mul121!(C::Union{Matrix,SubArray,Adjoint}, R::RkMatrix, M::Union{Matrix,SubArray,Adjoint}, a::Number)
    _mul111!(C, R.A, R.Bt * M, a)
end
function _mul121!(C::Union{Matrix,SubArray,Adjoint}, adjR::Adjoint{<:Any,<:RkMatrix}, M::Union{Matrix,SubArray,Adjoint}, a::Number)
    R   = LinearAlgebra.parent(adjR)
    tmp = adjoint(R.A) * M
    _mul111!(C, R.B, tmp, a)
    return C
end

################################################################################
## 1.2.2
################################################################################
function _mul122!(C::Union{Matrix,SubArray,Adjoint}, R::RkMatrix, S::RkMatrix, a::Number)
    if rank(R) < rank(S)
        _mul111!(C, R.A, (R.Bt * S.A) * S.Bt, a)
    else
        _mul111!(C, R.A * (R.Bt * S.A), S.Bt, a)
    end
    return C
end

################################################################################
## 1.2.3
################################################################################
function _mul123!(C::Union{Matrix,SubArray,Adjoint}, R::RkMatrix, H::HMatrix, a::Number)
    T = promote_type(eltype(R), eltype(H))
    tmp = zeros(T, size(R.Bt, 1), size(H, 2))
    _mul113!(tmp, R.Bt, H, 1)
    _mul111!(C, R.A, tmp, a)
    return C
end

################################################################################
## 1.3.1
################################################################################
function _mul131!(C::Union{Matrix,SubArray,Adjoint}, H::HMatrix, M::Union{Matrix,SubArray,Adjoint}, a::Number)
    if hasdata(H)
        mat = data(H)
        if mat isa Matrix
            _mul111!(C, mat, M, a)
        elseif mat isa RkMatrix
            _mul121!(C, mat, M, a)
        else
            error()
        end
    end
    for child in children(H)
        shift  = pivot(H) .- 1
        irange = rowrange(child) .- shift[1]
        jrange = colrange(child) .- shift[2]
        Cview  = view(C, irange, :)
        Mview  = view(M, jrange, :)
        _mul131!(Cview, child, Mview, a)
    end
    return C
end

################################################################################
## 1.3.2
################################################################################
function _mul132!(C::Union{Matrix,SubArray,Adjoint}, H::HMatrix, R::RkMatrix, a::Number)
    T = promote_type(eltype(H),eltype(R))
    buffer = zeros(T, size(H, 1), size(R.A, 2))
    _mul131!(buffer,H,R.A,1)
    _mul111!(C, buffer, R.Bt, a,)
    return C
end

################################################################################
## 1.3.3 (should never arise in practice, thus sloppy implementation)
################################################################################
function _mul133!(C::Union{Matrix,SubArray,Adjoint}, H::HMatrix, S::HMatrix, a::Number)
    @debug "1.3.3: this case should not arise"
    _mul131!(C,H,Matrix(S),a)
    return C
end

################################################################################
## 2.1.1
################################################################################
function _mul211!(C::RkMatrix, M::Union{Matrix,SubArray,Adjoint}, F::Union{Matrix,SubArray,Adjoint}, a::Number,compressor=identity)
    @debug "2.1.1: this case should not arise"
    T = promote_type(eltype(M),eltype(F))
    buffer = zeros(T,size(M,1),size(F,2))
    _mul111!(buffer,M,F,1)
    axpy!(a,buffer,C)
    compress!(C,compressor)
    return C
end

################################################################################
## 2.1.2
################################################################################
function _mul212!(C::RkMatrix, M::Union{Matrix,SubArray,Adjoint}, R::RkMatrix, a::Number,compressor=identity)
    tmp = RkMatrix(M * R.A, R.B)
    axpy!(a, tmp, C)
    compress!(C,compressor)
end

################################################################################
## 2.1.3
################################################################################
function _mul213!(C::RkMatrix, M::Union{Matrix,SubArray,Adjoint}, H::HMatrix, a::Number,compressor=identity)
    @debug "2.1.3: this case should not arise"
    T = promote_type(eltype(M),eltype(H))
    buffer = zeros(T,size(M,1),size(H,2))
    _mul113!(buffer,M,H,1)
    axpy!(a,buffer,C)
    compress!(C,compressor)
    return C
end

################################################################################
## 2.2.1
################################################################################
function _mul221!(C::RkMatrix, R::RkMatrix, M::Union{Matrix,SubArray,Adjoint}, a::Number,compressor=identity)
    tmp = RkMatrix(R.A, adjoint(M) * R.B)
    axpy!(a, tmp,C)
    compress!(C,compressor)
    return C
end

################################################################################
## 2.2.2
################################################################################
function _mul222!(C::RkMatrix, R::RkMatrix, S::RkMatrix, a::Number,compressor=identity)
    if rank(R) < rank(S)
        tmp = RkMatrix(R.A, S.B*(S.At*R.B))
    else
        tmp = RkMatrix(R.A*(R.Bt*S.A) , S.B)
    end
    axpy!(a, tmp,C)
    compress!(C,compressor)
    return C
end

################################################################################
## 2.2.3
################################################################################
function _mul223!(C::RkMatrix, R::RkMatrix, H::HMatrix, a::Number,compressor=identity)
    T      = promote_type(eltype(R), eltype(H))
    buffer = zeros(T, size(R.Bt, 1), size(H, 2))
    _mul113!(buffer, R.Bt, H, 1)
    # TODO: implement method to do mul!(buffer,adjoint(H),R.B) instead of the
    # above
    tmp = RkMatrix(R.A, collect(adjoint(buffer)))
    axpy!(a, tmp, C)
    compress!(C,compressor)
    return C
end

################################################################################
## 2.3.1
################################################################################
function _mul231!(C::RkMatrix, H::HMatrix, M::Union{Matrix,SubArray,Adjoint}, a::Number,compressor=identity)
    @debug "2.3.1: this case should not arise"
    T = promote_type(eltype(H),eltype(M))
    buffer = zeros(T,size(H,1),size(M,2))
    _mul131!(buffer,H,M,1)
    axpy!(a,buffer,C)
    compress!(C,compressor)
    return C
end

################################################################################
## 2.3.2
################################################################################
function _mul232!(C::RkMatrix, H::HMatrix, R::RkMatrix, a::Number,compressor=identity)
    T = promote_type(eltype(H), eltype(R))
    buffer = zeros(T, size(H, 1), size(R.A, 2))
    _mul131!(buffer, H, R.A,1)
    tmp = RkMatrix(buffer, R.B)
    axpy!(a, tmp, C)
    compress!(C,compressor)
    return C
end

################################################################################
## 2.3.3
################################################################################
# This case should be studied further. A few easy solutions are possible:
# (a) convert one of the HMatrices to RkMatrix (requires converting the full blocks to rkmatrix using say svd)
# (b) convert one of the HMatrices to Matrix   (requires converting the sparse blocks to full using outer product)
# (c) recurse down the  Hmatrices structure
# TODO: benchmark and test

# option b
# function _mul233!(C::RkMatrix, A::HMatrix, B::HMatrix, a::Number,compressor=identity)
#     if length(A) < length(B)
#         return _mul213!(C,Matrix(A),B,a)
#     else
#         return _mul231!(C,A,Matrix(B),a)
#     end
# end

# option c
function _mul233!(C::RkMatrix, A::HMatrix, B::HMatrix, a::Number,compressor=identity)
    T = promote_type(eltype(A), eltype(B))
    chdA = children(A)
    chdB = children(B)
    # if neither A nor B is a leaf, recurse on their children creating an
    # RkMatrix on each step, then aggregate this matrix of RkMatrices.
    if !isleaf(A) && !isleaf(B)
        m, n    = size(chdA, 1), size(chdB,2)
        k       = size(chdA,2)
        block  = Matrix{typeof(C)}(undef, m, n) # block of RkMatrices
        for i = 1:m
            for j = 1:n
                p = size(chdA[i, 1], 1)
                q = size(chdB[1, j], 2)
                block[i,j] = RkMatrix(zeros(T,p,0),zeros(T,q,0))
                for l = 1:k
                    _mul233!(block[i,j], chdA[i, l], chdB[l, j], true, compressor)
                end
            end
        end
        R = aggregate(block,compressor)
        axpy!(a, R, C)
        compress!(R,compressor)
    else
        # terminal case. Sort the data and dispatch to other method
        Adata = isleaf(A) ? A.data : A
        Bdata = isleaf(B) ? B.data : B
        if Adata isa HMatrix
            if Bdata isa Matrix
                _mul231!(C, Adata, Bdata, a,compressor)
            elseif Bdata isa RkMatrix
                _mul232!(C, Adata, Bdata, a,compressor)
            end
        elseif Adata isa Matrix
            if Bdata isa Matrix
                _mul211!(C, Adata, Bdata, a,compressor)
            elseif Bdata isa RkMatrix
                _mul212!(C, Adata, Bdata, a,compressor)
            elseif Bdata isa HMatrix
                _mul213!(C, Adata, Bdata, a,compressor)
            end
        elseif Adata isa RkMatrix
            if Bdata isa Matrix
                _mul221!(C, Adata, Bdata, a,compressor)
            elseif Bdata isa RkMatrix
                _mul222!(C, Adata, Bdata, a,compressor)
            elseif Bdata isa HMatrix
                _mul223!(C, Adata, Bdata, a,compressor)
            end
        end
    end
    return C
end

function aggregate(B::Matrix{<:RkMatrix},compressor)
    @assert size(B) == (2,2)
    B1 = hcat(B[1,1],B[1,2])
    compress!(B1,compressor)
    B2 = hcat(B[2,1],B[2,2])
    compress!(B2,compressor)
    return compress!(vcat(B1,B2),compressor)
end

################################################################################
## 3.1.1
################################################################################
function _mul311!(C::HMatrix,M::Union{Matrix,SubArray,Adjoint},F::Union{Matrix,SubArray,Adjoint},a::Number)
    tmp = a*M*F
    if hasdata(C)
        axpy!(true,tmp,data(C))
    else
        C.data = tmp
    end
    return C
end

################################################################################
## 3.1.2
################################################################################
function _mul312!(C::HMatrix,M::Union{Matrix,SubArray,Adjoint},R::RkMatrix,a::Number,compressor=identity)
    buffer = a*M*R.A
    tmp    = RkMatrix(buffer,copy(R.B))
    if hasdata(C)
        axpy!(true,tmp,C.data)
    else
        C.data = tmp
    end
    compress!(C.data,compressor)
    return C
end

################################################################################
## 3.1.3
################################################################################
function _mul313!(C::HMatrix,M::Union{Matrix,SubArray,Adjoint},H::HMatrix,a::Number)
    @debug "3.1.3: this case should not arise"
    T = promote_type(eltype(M),eltype(H))
    buffer = zeros(T,size(M,1),size(H,2))
    _mul113!(buffer,M,H,a)
    if hasdata(C)
        axpy!(true,buffer,C.data)
    else
        C.data = buffer
    end
    return C
end

################################################################################
## 3.2.1
################################################################################
function _mul321!(C::HMatrix,R::RkMatrix,M::Union{Matrix,SubArray,Adjoint},a::Number,compressor=identity)
    buff = conj(a)*adjoint(M)*R.B
    tmp  = RkMatrix(copy(R.A),buff)
    if hasdata(C)
        axpy!(true,tmp,C.data)
    else
        C.data = tmp
    end
    compress!(C.data,compressor)
    return C
end

################################################################################
## 3.2.2
################################################################################
function _mul322!(C::HMatrix,R::RkMatrix,S::RkMatrix,a::Number,compressor=identity)
    if rank(R) < rank(S)
        tmp = RkMatrix(copy(R.A), conj(a)*S.B*(S.At*R.B))
    else
        tmp = RkMatrix(a*R.A*(R.Bt*S.A) , copy(S.B))
    end
    if hasdata(C)
        axpy!(true,tmp,C.data)
    else
        C.data = tmp
    end
    compress!(C.data,compressor)
    return C
end

################################################################################
## 3.2.3
################################################################################
function _mul323!(C::HMatrix,R::RkMatrix,H::HMatrix,a::Number,compressor=identity)
    T = promote_type(eltype(R.Bt),eltype(H))
    buffer = zeros(T,size(R.Bt,1),size(H,2))
    _mul113!(buffer,R.Bt,H,a)
    tmp = RkMatrix(copy(R.A),collect(adjoint(buffer)))
    if hasdata(C)
        axpy!(true,tmp,C.data)
    else
        C.data = tmp
    end
    compress!(C.data,compressor)
    return C
end

################################################################################
## 3.3.1
################################################################################
function _mul331!(C::HMatrix,H::HMatrix,M::Matrix,a::Number)
    @debug "3.3.1: this case should not arise"
    T = promote_type(eltype(H),eltype(M))
    buffer = zeros(T,size(H,1),size(M,2))
    _mul131!(buffer,H,M,a)
    if hasdata(C)
        axpy!(true,buffer,C.data)
    else
        C.data = buffer
    end
    return C
end

################################################################################
## 3.3.2
################################################################################
function _mul332!(C::HMatrix,H::HMatrix,R::RkMatrix,a::Number,compressor=identity)
    T = promote_type(eltype(H),eltype(R))
    buffer = zeros(T,size(H,1),size(R.A,2))
    _mul131!(buffer,H,R.A,a)
    tmp = RkMatrix(buffer,copy(R.B))
    if hasdata(C)
        axpy!(true,tmp,C.data)
    else
        C.data = tmp
    end
    compress!(C.data,compressor)
    return C
end

################################################################################
## 3.3.3 (the recursive case)
################################################################################
function _mul333!(C::HMatrix,A::HMatrix,B::HMatrix,a::Number,compressor=identity)
    # when either A, B, or C is a leaf, extract the data and dispatch to one of the other
    # mul! methods above. Otherwise recurse on children.
    if isleaf(A) || isleaf(B) || isleaf(C)
        @timeit_debug "leaf multiplication" begin
            _mul_leaf!(C,A,B,a,compressor)
        end
    else
        ni,nj = blocksize(C)
        _ ,nk = blocksize(A)
        A_children = children(A)
        B_children = children(B)
        C_children = children(C)
        for i=1:ni
            for j=1:nj
                for k=1:nk
                    _mul333!(C_children[i,j],A_children[i,k],B_children[k,j],a,compressor)
                end
            end
        end
    end
    return C
end

# terminal case of the multiplication when at least one of the arguments is a leaf
function _mul_leaf!(C::HMatrix,A::HMatrix,B::HMatrix,a::Number,compressor)
    # possible types of arguments after extracting data
    T = eltype(C)
    Mat = Matrix{T}
    RkMat = RkMatrix{T}
    HMat = typeof(C)

    @assert isleaf(A) || isleaf(B) || isleaf(C)
    Cdata  = hasdata(C) ? data(C) : C
    Adata  = hasdata(A) ? data(A) : A
    Bdata  = hasdata(B) ? data(B) : B

    # no compression needed
    if Cdata isa Matrix
        if Adata isa Matrix
            if Bdata isa Matrix
                _mul111!(Cdata::Mat,Adata::Mat,Bdata::Mat,a)
            elseif Bdata isa RkMatrix
                _mul112!(Cdata::Mat,Adata::Mat,Bdata::RkMat,a)
            elseif Bdata isa HMatrix
                _mul113!(Cdata::Mat,Adata::Mat,Bdata::HMat,a)
            end
        elseif Adata isa RkMatrix
            if Bdata isa Matrix
                _mul121!(Cdata::Mat,Adata::RkMat,Bdata::Mat,a)
            elseif Bdata isa RkMatrix
                _mul122!(Cdata::Mat,Adata::RkMat,Bdata::RkMat,a)
            elseif Bdata isa HMatrix
                _mul123!(Cdata::Mat,Adata::RkMat,Bdata::RkMat,a)
            end
        elseif Adata isa HMatrix
            if Bdata isa Matrix
                _mul131!(Cdata::Mat,Adata::HMat,Bdata::Mat,a)
            elseif Bdata isa RkMatrix
                _mul132!(Cdata::Mat,Adata::HMat,Bdata::RkMat,a)
            elseif Bdata isa HMatrix
                _mul133!(Cdata::Mat,Adata::HMat,Bdata::HMat,a)
            end
        end
    elseif Cdata isa RkMatrix # compress after multiplication
        if Adata isa Matrix
            if Bdata isa Matrix
                _mul211!(Cdata::RkMat,Adata::Mat,Bdata::Mat,a,compressor)
            elseif Bdata isa RkMatrix
                _mul212!(Cdata::RkMat,Adata::Mat,Bdata::RkMat,a,compressor)
            elseif Bdata isa HMatrix
                _mul213!(Cdata::RkMat,Adata::Mat,Bdata::HMat,a,compressor)
            end
        elseif Adata isa RkMatrix
            if Bdata isa Matrix
                _mul221!(Cdata::RkMat,Adata::RkMat,Bdata::Mat,a,compressor)
            elseif Bdata isa RkMatrix
                _mul222!(Cdata::RkMat,Adata::RkMat,Bdata::RkMat,a,compressor)
            elseif Bdata isa HMatrix
                _mul223!(Cdata::RkMat,Adata::RkMat,Bdata::HMat,a,compressor)
            end
        elseif Adata isa HMatrix
            if Bdata isa Matrix
                _mul231!(Cdata::RkMat,Adata::HMat,Bdata::Mat,a,compressor)
            elseif Bdata isa RkMatrix
                _mul232!(Cdata::RkMat,Adata::HMat,Bdata::RkMat,a,compressor)
            elseif Bdata isa HMatrix
                _mul233!(Cdata::RkMat,Adata::HMat,Bdata::HMat,a,compressor)
            end
        end
    elseif Cdata isa HMatrix # compress and flush to leaves
        if Adata isa Matrix
            if Bdata isa Matrix
                _mul311!(Cdata::HMat,Adata::Mat,Bdata::Mat,a)
            elseif Bdata isa RkMatrix
                _mul312!(Cdata::HMat,Adata::Mat,Bdata::RkMat,a,compressor)
            elseif Bdata isa HMatrix
                _mul313!(Cdata::HMat,Adata::Mat,Bdata::HMat,a)
            end
        elseif Adata isa RkMatrix
            if Bdata isa Matrix
                _mul321!(Cdata::HMat,Adata::RkMat,Bdata::Mat,a,compressor)
            elseif Bdata isa RkMatrix
                _mul322!(Cdata::HMat,Adata::RkMat,Bdata::RkMat,a,compressor)
            elseif Bdata isa HMatrix
                _mul323!(Cdata::HMat,Adata::RkMat,Bdata::HMat,a,compressor)
            end
        elseif Adata isa HMatrix
            if Bdata isa Matrix
                _mul331!(Cdata::HMat,Adata::HMat,Bdata::Mat,a)
            elseif Bdata isa RkMatrix
                _mul332!(Cdata::HMat,Adata::HMat,Bdata::RkMat,a,compressor)
            end
        end
        flush_to_leaves!(C,compressor)
    end
    return C
end

compress!(data::RkMatrix,::typeof(identity)) = data
compress!(data::Matrix,::typeof(identity))   = data
compress!(data::HMatrix,::typeof(identity))   = data

"""
    flush_to_leaves(H::HMatrix)

Similar to [`transfer_to_children`](@ref), but transfer the data from `H` all
the way down to its leaves.
"""
function flush_to_leaves!(H::HMatrix,compressor)
    hasdata(H) && !isleaf(H) || (return H)
    R = data(H)
    _add_to_leaves!(H,R,compressor)
    H.data = nothing
    return H
end

function _add_to_leaves!(H::HMatrix,R::RkMatrix,compressor)
    shift = pivot(H) .- 1
    for block in Leaves(H)
        irange     = rowrange(block) .- shift[1]
        jrange     = colrange(block) .- shift[2]
        bdata      = data(block)
        tmp        = RkMatrix(R.A[irange,:],R.B[jrange,:])
        if bdata === nothing
            setdata!(block,tmp)
        else
            axpy!(true,tmp,bdata)
            bdata isa RkMatrix && compress!(bdata,compressor)
        end
    end
    return H
end

############################################################################################
# Specializations on gemv:
# The routines below provide specialized version of mul!(C,A,B,a,b) when `A` and
# `B` are vectors
############################################################################################

# 1.2.1
function LinearAlgebra.mul!(y::AbstractVector,R::RkMatrix,x::AbstractVector,a::Number,b::Number)
    tmp = R.Bt*x
    mul!(y,R.A,tmp,a,b)
end

# 1.2.1
function LinearAlgebra.mul!(y::AbstractVector,adjR::Adjoint{<:Any,<:RkMatrix},x::AbstractVector,a::Number,b::Number)
    R = parent(adjR)
    tmp = R.At*x
    mul!(y,R.B,tmp,a,b)
end

# 1.3.1
function LinearAlgebra.mul!(y::AbstractVector,A::HMatrix,x::AbstractVector,a::Number,b::Number;
                            global_index=true,threads=true,distributed=false)
    # since the HMatrix represents A = Pr*H*Pc, where Pr and Pc are row and column
    # permutations, we need first to rewrite C <-- b*C + a*(Pc*H*Pb)*B as
    # C <-- Pr*(b*inv(Pr)*C + a*H*(Pc*B)). Following this rewrite, the
    # multiplication is performed by first defining B <-- Pc*B, and C <--
    # inv(Pr)*C, doing the multiplication with the permuted entries, and then
    # permuting the result  C <-- Pr*C at the end.
    ctree     = A.coltree
    rtree     = A.rowtree
    # permute input
    if global_index
        x         = x[ctree.loc2glob]
        y         = permute!(y,rtree.loc2glob)
        rmul!(x,a) # multiply in place since this is a new copy, so does not mutate exterior x
    else
        x = a*x # new copy of x
    end
    rmul!(y,b)
    # offset in case A is not indexed starting at (1,1); e.g. A is not the root
    # of and HMatrix
    offset = pivot(A) .- 1
    if threads
        # TODO: test the various threaded implementations and chose one.
        # Currently there are two main choices:
        # 1. spawn a task per leaf, and let julia scheduler handle the tasks
        # 2. create a static partition of the leaves and try to estimate the
        #    cost, then spawn one task per block of the partition. In this case,
        #    test if the hilbert partition is really faster than col_partition
        #    or row_partition
        # Right now the hilbert partition is chosen by default without proper testing.
        nt        = Threads.nthreads()
        partition = hilbert_partitioning(A,nt)
        _hgemv_static_partition!(y,x,partition,offset)
        # _hgemv_threads!(y,A,x,offset)  # threaded implementation
    else
        _hgemv_recursive!(y,A,x,offset) # serial implementation
    end
    # permute output
    global_index && invpermute!(y,loc2glob(rtree))
    return y
end

"""
    _hgemv_recursive!(C,A,B,offset)

Internal function used to compute `C[I] <-- C[I] + A*B[J]` where `I =
rowrange(A) - offset[1]` and `J = rowrange(B) - offset[2]`.

The `offset` argument is used on the caller side to signal if the original
hierarchical matrix had a `pivot` other than `(1,1)`.
"""
function _hgemv_recursive!(C::AbstractVector,A::Union{HMatrix,Adjoint{<:Any,<:HMatrix}},B::AbstractVector,offset)
    T = eltype(A)
    if isleaf(A)
        irange = rowrange(A) .- offset[1]
        jrange = colrange(A) .- offset[2]
        d   = data(A)
        if T <: Number
            # C and B are the "global" vectors handled by the caller, so a view is needed.
            LinearAlgebra.mul!(view(C,irange),d,view(B,jrange),1,1)
        elseif T<: SMatrix
            # FIXME: there is bug with gemv and static arrays, so we convert
            # them to matrices of n × 1
            if d isa Matrix
                LinearAlgebra.mul!(view(C,irange,1:1),d,view(B,jrange,1:1),1,1)
            else
                LinearAlgebra.mul!(view(C,irange),d,view(B,jrange),1,1)
            end
        else
            error("T=$T")
        end
    else
        for block in children(A)
            _hgemv_recursive!(C,block,B,offset)
        end
    end
    return C
end

function _hgemv_threads!(C::AbstractVector,A::HMatrix,B::AbstractVector,offset)
    # make copies of C and run in parallel
    nt        = Threads.nthreads()
    Cthreads  = [zero(C) for _ in 1:nt]
    blocks    = Leaves(A)
    @sync for block in blocks
        Threads.@spawn begin
            id = Threads.threadid()
            _hgemv_recursive!(Cthreads[id],block,B,offset)
        end
    end
    # reduce
    for Ct in Cthreads
        axpy!(1,Ct,C)
    end
    return C
end

function _hgemv_static_partition!(C::AbstractVector,B::AbstractVector,partition,offset)
    # multiply by b at root level
    # rmul!(C,b)
    # create a lock for the reduction step
    mutex = ReentrantLock()
    nt    = length(partition)
    times = zeros(nt)
    Threads.@threads for n in 1:nt
        id = Threads.threadid()
        times[id] =
        @elapsed begin
            leaves = partition[n]
            Cloc   = zero(C)
            for leaf in leaves
                irange = rowrange(leaf) .- offset[1]
                jrange = colrange(leaf) .- offset[2]
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
    leaves = Leaves(H) |> collect
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

# TODO: benchmark the different partitioning strategies for gemv. Is the hilber
# partition really faster than the simpler alternatives (row partition, col partition)?

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

"""
    cost_mv(A::Union{Matrix,SubArray,Adjoint})

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

############################################################################################
####################################### rmul! ##############################################
############################################################################################
function LinearAlgebra.rmul!(R::RkMatrix, b::Number)
    m, n = size(R)
    if m > n
        rmul!(R.B, conj(b))
    else
        rmul!(R.A, b)
    end
    return R
end

function LinearAlgebra.rmul!(H::HMatrix, b::Number)
    b == true && (return H) # short circuit. If inlined, rmul!(H,true) --> no-op
    if hasdata(H)
        rmul!(data(H), b)
    end
    for child in children(H)
        rmul!(child, b)
    end
    return H
end
