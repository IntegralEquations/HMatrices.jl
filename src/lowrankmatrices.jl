"""
    RkMatrix{T}

Representation of a rank `r` matrix ``M`` in an outer product format `M =
A*adjoint(B)` where `A` has size `m × r` and `B` has size `n × r`.

The internal representation stores `A` and `B`, but `R.Bt` or `R.At` can be used
to get the respective adjoints.
"""
struct RkMatrix{T}
    A::Matrix{T}
    B::Matrix{T}
    buffer::Vector{T}
    function RkMatrix(A::Matrix{T},B::Matrix{T}) where {T<:Number}
        @assert size(A,2) == size(B,2) "second dimension of `A` and `B` must match"
        m,r = size(A)
        n  = size(B,1)
        if  r*(m+n) >= m*n
            @debug "Inefficient RkMatrix:" size(A) size(B)
        end
        buffer = Vector{T}(undef,r)
        new{T}(A,B,buffer)
    end
end
RkMatrix(A,B) = RkMatrix(promote(A,B)...)

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
    vcat(M1::RkMatrix,M2::RkMatrix)

`RkMatrix` can be concatenated horizontally or vertically to produce a new
`RkMatrix` of rank `rank(M1)+rank(M2)`
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

function LinearAlgebra.mul!(C::AbstractVector,Rk::RkMatrix,F::AbstractVector,a::Number,b::Number)
    mul!(Rk.buffer,adjoint(Rk.B),F)
    mul!(C,Rk.A,Rk.buffer,a,b)
end

function Base.:(*)(R::RkMatrix{T},x::AbstractVector{S}) where {T,S}
    TS = promote_type(T,S)
    y  = Vector{TS}(undef,size(R,1))
    mul!(y,R,x)
end

Base.rand(::Type{RkMatrix{T}},m::Int,n::Int,r::Int) where {T} = RkMatrix(rand(T,m,r),rand(T,n,r)) #useful for testing
Base.rand(::Type{RkMatrix},m::Int,n::Int,r::Int) = rand(RkMatrix{Float64},m,n,r)

Base.copy(R::RkMatrix) = RkMatrix(copy(R.A),copy(R.B))

Base.Matrix(R::RkMatrix)    = R.A*R.Bt

"""
    num_elements(R::RkMatrix)

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
