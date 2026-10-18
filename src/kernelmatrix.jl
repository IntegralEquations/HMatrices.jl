"""
    abstract type AbstractKernelMatrix{T} <: AbstractMatrix{T}

Interface for abstract matrices represented through a kernel function `f`, target
elements `X`, and source elements `Y`. The matrix entry
`i,j` is given by `f(X[i],Y[j])`. Concrete subtypes should implement
`Base.size`, `Base.getindex(K, i::Int, j::Int)`, and [`rowelements`](@ref) and
[`colelements`](@ref), whose elements have a [`center`](@ref) used to cluster them.

Implementing [`getblock!`](@ref) (e.g. with SIMD vectorization) can improve the speed
of assembling an [`HMatrix`](@ref).
"""
abstract type AbstractKernelMatrix{T} <: AbstractMatrix{T} end

"""
    KernelMatrix{Tf,Tx,Ty,T} <:: AbstractKernelMatrix{T}

Generic kernel matrix representing a kernel function acting on two sets of
elements. If `K` is a `KernelMatrix`, then `K[i,j] = f(X[i],Y[j])` where
`f::Tf=kernel(K)`, `X::Tx=rowelements(K)` and `Y::Ty=colelements(K)`.

# Examples
```julia
X = rand(SVector{2,Float64},2)
Y = rand(SVector{2,Float64},2)
K = KernelMatrix(X,Y) do x,y
    sum(x+y)
end
```
"""
struct KernelMatrix{Tf, Tx, Ty, T} <: AbstractKernelMatrix{T}
    f::Tf
    X::Tx
    Y::Ty
end

Base.size(K::KernelMatrix) = length(K.X), length(K.Y)
Base.getindex(K::KernelMatrix, i::Int, j::Int) = K.f(K.X[i], K.Y[j])

"""
    rowelements(K::AbstractKernelMatrix)

The elements associated with the rows of `K`.
"""
rowelements(K::KernelMatrix) = K.X

"""
    colelements(K::AbstractKernelMatrix)

The elements associated with the columns of `K`.
"""
colelements(K::KernelMatrix) = K.Y
kernel(K::KernelMatrix) = K.f

function KernelMatrix(f, X, Y)
    T = Base.promote_op(f, eltype(X), eltype(Y))
    return KernelMatrix{typeof(f), typeof(X), typeof(Y), T}(f, X, Y)
end

"""
    assemble_hmatrix(K::AbstractKernelMatrix; kwargs...)
    assemble_hmatrix(Hermitian(K::AbstractKernelMatrix); kwargs...)

Approximate `K` by an [`HMatrix`](@ref), clustering the elements of `K` by their
[`center`](@ref). The keyword arguments are those of [`assemble_hmatrix`](@ref).
"""
function assemble_hmatrix(K::AbstractKernelMatrix; kwargs...)
    return assemble_hmatrix(K, rowelements(K), colelements(K); kwargs...)
end
function assemble_hmatrix(K::Hermitian{<:Any, <:AbstractKernelMatrix}; kwargs...)
    return assemble_hmatrix(K, rowelements(parent(K)); kwargs...)
end
