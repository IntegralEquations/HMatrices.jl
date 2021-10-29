"""
    mutable struct RkMatrix{T}

Representation of a rank `r` matrix `M` in an outer product format `M =
A*adjoint(B)` where `A` has size `m × r` and `B` has size `n × r`.

The internal representation stores `A` and `B`, but `R.Bt` or `R.At` can be used
to get the respective adjoints.
"""
mutable struct RkMatrix{T} <: AbstractMatrix{T}
    A::Matrix{T}
    B::Matrix{T}
    function RkMatrix(A::Matrix{T},B::Matrix{T}) where {T}
        @assert size(A,2) == size(B,2) "second dimension of `A` and `B` must match"
        m,r = size(A)
        n  = size(B,1)
        if  r*(m+n) >= m*n
            @debug "Inefficient RkMatrix:" size(A) size(B)
            # error("Inefficient RkMatrix")
        end
        new{T}(A,B)
    end
end
RkMatrix(A,B) = RkMatrix(promote(A,B)...)

function Base.getindex(rmat::RkMatrix,i::Int,j::Int)
    @debug "calling slow `getindex` of an `RkMatrix`"
    # error("calling slow `getindex` of an `RkMatrix`")
    r = rank(rmat)
    acc = zero(eltype(rmat))
    for k in 1:r
        acc += rmat.A[i,k] * conj(rmat.B[j,k])
    end
    return acc
end

# some "fast" ways of computing a row and column of R and ajoint(R)
function Base.getindex(R::RkMatrix, ::Colon, j::Int)
    R.A*conj(view(R.B,j,:))
end

# return the j-th column
function Base.getindex(R::RkMatrix, i::Int, ::Colon)
    # conj(R.B)*view(R.A,i,:)
    R.B*conj(view(R.A,i,:)) |> conj!
end

function Base.getindex(Ra::Adjoint{<:Any,<:RkMatrix}, i::Int, ::Colon)
    R = LinearAlgebra.parent(Ra)
    R.A*conj(view(R.B,i,:)) |> conj!
end

function Base.getindex(Ra::Adjoint{<:Any,<:RkMatrix}, ::Colon, j::Int)
    R = LinearAlgebra.parent(Ra)
    R.B*conj(view(R.A,j,:))
end

"""
    RkMatrix(A::Vector{<:Vector},B::Vector{<:Vector})

Construct an `RkMatrix` from a vector of vectors. Assumes that `length(A) ==
length(B)`, which determines the rank, and that all vectors in `A` (resp. `B`) have the same length `m`
(resp. `n`).
"""
function RkMatrix(_A::Vector{V},_B::Vector{V}) where {V<:AbstractVector}
    T   = eltype(V)
    @assert length(_A) == length(_B)
    k   = length(_A)
    m   = length(first(_A))
    n   = length(first(_B))
    A   = Matrix{T}(undef,m,k)
    B  = Matrix{T}(undef,n,k)
    for i in 1:k
        copyto!(view(A,:,i),_A[i])
        copyto!(view(B,:,i),_B[i])
    end
    return RkMatrix(A,B)
end

"""
    RkMatrix(F::LinearAlgebra.SVD)
    RkMatrix!(F::LinearAlgebra.SVD)

Construct an [`RkMatrix`](@ref) from an `SVD` factorization. The `!` version
invalidates the data in `F`.
"""
function RkMatrix(F::LinearAlgebra.SVD)
    A  = F.U*LinearAlgebra.Diagonal(F.S)
    B  = copy(F.V)
    return RkMatrix(A,B)
end
function RkMatrix!(F::LinearAlgebra.SVD)
    A  = rmul!(F.U,LinearAlgebra.Diagonal(F.S))
    B  = F.V
    return RkMatrix(A,B)
end

Base.eltype(::RkMatrix{T}) where {T} = T
Base.size(rmat::RkMatrix)                                        = (size(rmat.A,1), size(rmat.B,1))
Base.size(rmat::RkMatrix,i)                                      = size(rmat)[i]
Base.length(rmat::RkMatrix)                                      = prod(size(rmat))
Base.isapprox(rmat::RkMatrix,B::AbstractArray,args...;kwargs...) = isapprox(Matrix(rmat),B,args...;kwargs...)

LinearAlgebra.rank(M::RkMatrix) = size(M.A,2)

function Base.getproperty(R::RkMatrix,s::Symbol)
    if  s == :Bt
        return adjoint(R.B)
    elseif  s == :At
        return adjoint(R.A)
    else
        return getfield(R,s)
    end
end

"""
    hcat(M1::RkMatrix,M2::RkMatrix)

Concatenated `M1` and `M2` horizontally to produce a new `RkMatrix` of rank
`rank(M1)+rank(M2)`
"""
function Base.hcat(M1::RkMatrix{T},M2::RkMatrix{T}) where {T}
    m,n  = size(M1)
    s,t  = size(M2)
    (m == s) || throw(ArgumentError("number of rows of each array must match: got  ($m,$s)"))
    r1   = size(M1.A,2)
    r2   = size(M2.A,2)
    A    = hcat(M1.A,M2.A)
    B1   = vcat(M1.B,zeros(T,t,r1))
    B2   = vcat(zeros(T,n,r2),M2.B)
    B    = hcat(B1,B2)
    return RkMatrix(A,B)
end

"""
    vcat(M1::RkMatrix,M2::RkMatrix)

Concatenated `M1` and `M2` vertically to produce a new `RkMatrix` of rank
`rank(M1)+rank(M2)`
"""
function Base.vcat(M1::RkMatrix{T},M2::RkMatrix{T}) where {T}
    m,n  = size(M1)
    s,t  = size(M2)
    n == t || throw(ArgumentError("number of columns of each array must match (got  ($n,$t))"))
    r1   = size(M1.A,2)
    r2   = size(M2.A,2)
    A1   = vcat(M1.A,zeros(T,s,r1))
    A2   = vcat(zeros(T,m,r2),M2.A)
    A    = hcat(A1,A2)
    B    = hcat(M1.B,M2.B)
    return RkMatrix(A,B)
end

# function LinearAlgebra.mul!(C::AbstractVector,Rk::RkMatrix{T},F::AbstractVector,a::Number,b::Number) where {T<:SMatrix}
#     buf = buffer(Rk)
#     m,n = size(Rk)
#     r   = rank(Rk)
#     # mul!(buf,Rk.Bt,F)
#     for k in 1:r
#         buf[k] = zero(T)
#         for j in 1:n
#             buf[k] += Rk.B[j,k]*F[j]
#         end
#     end
#     # mul!(C,Rk.A,buf,a,b)
#     for i in 1:m
#         C[i] = b*C[i]
#         for k in 1:r
#             C[i] += Rk.A[i,k]*buf[k]
#         end
#     end
#     return C[i]
# end

function Base.:(*)(R::RkMatrix{T},x::AbstractVector{S}) where {T,S}
    TS = promote_type(T,S)
    y  = Vector{TS}(undef,size(R,1))
    mul!(y,R,x)
end

Base.rand(::Type{RkMatrix{T}},m::Int,n::Int,r::Int) where {T} = RkMatrix(rand(T,m,r),rand(T,n,r)) #useful for testing
Base.rand(::Type{RkMatrix},m::Int,n::Int,r::Int) = rand(RkMatrix{Float64},m,n,r)

Base.copy(R::RkMatrix) = RkMatrix(copy(R.A),copy(R.B))

"""
    num_stored_elements(R::RkMatrix)

The number of entries stored in the representation. Note that this is *not*
`size(R)`.
"""
num_stored_elements(R::RkMatrix)        = size(R.A,2)*(sum(size(R)))

"""
    compression_ratio(R::RkMatrix)

The ratio of the uncompressed size of `R` to its compressed size in outer
product format.
"""
compression_ratio(R::RkMatrix) = prod(size(R)) / num_stored_elements(R)

function LinearAlgebra.mul!(C::AbstractVector,Rk::RkMatrix{T},F::AbstractVector,a::Number,b::Number) where {T<:SMatrix}
    m,n = size(Rk)
    r   = rank(Rk)
    tmp = Rk.Bt*F
    # FIXME: to support scalar and tensorial problems, we currently allow for T
    # to be something other than a plain number. If that is the case, we
    # implement a (slow) multiplication algorithm by hand to circumvent a
    # problem in LinearAlgebra for the generic mulplication mul!(C,A,B,a,b) when
    # C and B are a vectors of static matrices, and A is a matrix of static
    # matrices. Should eventually be removed.
    rmul!(C,b)
    for k in 1:r
        tmp[k] *= a
        for i in 1:m
            C[i] += Rk.A[i,k]*tmp[k]
        end
    end
    return C
end

"""
    svd(R::RkMatrix,[tol])

Compute the singular value decomposition of an `RkMatrix` by first doing a `qr`
of `R.A` and `R.B` followed by an `svd` of ``R_A*(R_{B})^T``. If passed `tol`,
discard all singular values smaller than `tol`.
"""
function LinearAlgebra.svd(R::RkMatrix)
    r      = rank(R)
    # treat weird case where it would be most efficient to convert first to a full matrix
    r > min(size(R)...) && return svd(Matrix(R))
    # qr part
    QA, RA = qr(R.A)
    QB, RB = qr(R.B)
    # svd part
    F      = svd(RA*adjoint(RB))
    # build U and Vt
    U      = QA*F.U
    Vt     = F.Vt*adjoint(QB)
    return SVD(U,F.S,Vt) #create the SVD structure
end
function LinearAlgebra.svd(A::RkMatrix{T},tol) where {T}
    F = svd(A)
    r = findlast(x -> x>tol, F.S)
    return SVD(F.U[:,1:r],F.S[1:r],F.V[:,1:r])
end

"""
    svd!(R::RkMatrix,[tol])

Like `svd`, but performs the intermediate `qr` in-place mutating the data in
`R`. You should not reuse `R` after calling this method.
"""
function LinearAlgebra.svd!(M::RkMatrix)
    r      = rank(M)
    QA, RA = qr!(M.A)
    QB, RB = qr!(M.B)
    F      = svd!(RA*adjoint(RB)) # svd of an r×r problem
    U      = QA*F.U
    Vt     = F.Vt*adjoint(QB)
    return SVD(U,F.S,Vt) #create the SVD structure
end
function LinearAlgebra.svd!(R::RkMatrix,tol)
    tol <= 0 && return R
    m,n    = size(R)
    F      = svd!(R)
    r      = findlast(x -> x>tol, F.S)
    SVD(F.U[:,1:r],F.S[1:r],F.V[:,1:r])
end
