"""
    mutable struct RkMatrix{T}

Representation of a rank `r` matrix `M` in outer product format `M =
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

# some "fast" ways of computing a column of R and row of ajoint(R)
Base.getindex(R::RkMatrix, ::Colon, j::Int) = getcol(R,j)

"""
    getcol!(col,M::AbstractMatrix,j)

Fill the entries of `col` with column `j` of `M`.
"""
function getcol!(col,R::RkMatrix,j::Int)
    mul!(col,R.A,conj(view(R.B,j,:)))
end

"""
    getcol(M::AbstractMatrix,j)

Return a vector containing the `j`-th column of `M`.
"""
function getcol(R::RkMatrix,j::Int)
    m = size(R,1)
    T = eltype(R)
    col = zeros(T,m)
    getcol!(col,R,j)
end

function getcol!(col,Ra::Adjoint{<:Any,<:RkMatrix},j::Int)
    R = parent(Ra)
    mul!(col,R.B,conj(view(R.A,j,:)))
end

function getcol(Ra::Adjoint{<:Any,<:RkMatrix},j::Int)
    m = size(Ra,1)
    T = eltype(Ra)
    col = zeros(T,m)
    getcol!(col,Ra,j)
end

Base.getindex(Ra::Adjoint{<:Any,<:RkMatrix}, ::Colon, j::Int) = getcol(Ra,j)

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

Base.eltype(::RkMatrix{T}) where {T} = T
Base.size(rmat::RkMatrix)                                        = (size(rmat.A,1), size(rmat.B,1))
Base.length(rmat::RkMatrix)                                      = prod(size(rmat))
Base.isapprox(rmat::RkMatrix,B::AbstractArray,args...;kwargs...) = isapprox(Matrix(rmat),B,args...;kwargs...)

rank(M::RkMatrix) = size(M.A,2)

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
`rank(M1)+rank(M2)`.
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

Base.copy(R::RkMatrix) = RkMatrix(copy(R.A),copy(R.B))

function Base.Matrix(R::RkMatrix{<:Number})
    Matrix(R.A*R.Bt)
end
function Base.Matrix(R::RkMatrix{<:SMatrix})
    # collect must be used when we have a matrix of `SMatrix` because of this issue:
    # https://github.com/JuliaArrays/StaticArrays.jl/issues/966#issuecomment-943679214
    R.A*collect(R.Bt)
end

"""
    RkMatrix(F::SVD)

Construct an [`RkMatrix`](@ref) from an `SVD` factorization.
"""
function RkMatrix(F::SVD)
    A  = F.U*Diagonal(F.S)
    B  = copy(F.V)
    return RkMatrix(A,B)
end

"""
    RkMatrix(M::Matrix)

Construct an [`RkMatrix`](@ref) from a `Matrix` by passing through the full
`svd` of `M`.
"""
function RkMatrix(M::Matrix)
    F = svd(M)
    RkMatrix(F)
end

"""
    num_stored_elements(R::RkMatrix)

The number of entries stored in the representation. Note that this is *not*
`length(R)`.
"""
num_stored_elements(R::RkMatrix)        = size(R.A,2)*(sum(size(R)))

"""
    compression_ratio(R::RkMatrix)

The ratio of the uncompressed size of `R` to its compressed size in outer
product format.
"""
compression_ratio(R::RkMatrix) = prod(size(R)) / num_stored_elements(R)

# FIXME: to support scalar and tensorial problems, we currently allow for T
# to be something other than a plain number. If that is the case, we
# implement a (slow) multiplication algorithm by hand to circumvent a
# problem in LinearAlgebra for the generic mulplication mul!(C,A,B,a,b) when
# C and B are a vectors of static matrices, and A is a matrix of static
# matrices. Should eventually be removed.
function mul!(C::AbstractVector,Rk::RkMatrix{T},F::AbstractVector,a::Number,b::Number) where {T<:SMatrix}
    m,n = size(Rk)
    r   = rank(Rk)
    tmp = Rk.Bt*F
    rmul!(C,b)
    for k in 1:r
        tmp[k] *= a
        for i in 1:m
            C[i] += Rk.A[i,k]*tmp[k]
        end
    end
    return C
end
