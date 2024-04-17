"""
    abstract type AbstractKernelMatrix{T} <: AbstractMatrix{T}

Interface for abstract matrices represented through a kernel function `f`, target
elements `X`, and source elements `Y`. The matrix entry
`i,j` is given by `f(X[i],Y[j])`. Concrete subtypes should implement at least

    `Base.getindex(K::AbstractKernelMatrix,i::Int,j::Int)`

If a more efficient implementation of `getindex(K,I::UnitRange,I::UnitRange)`,
`getindex(K,I::UnitRange,j::Int)` and `getindex(adjoint(K),I::UnitRange,j::Int)`
is available (e.g. with SIMD vectorization), implementing such methods can
improve the speed of assembling an [`HMatrix`](@ref).
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
struct KernelMatrix{Tf,Tx,Ty,T} <: AbstractKernelMatrix{T}
    f::Tf
    X::Tx
    Y::Ty
end

Base.size(K::KernelMatrix) = length(K.X), length(K.Y)
Base.getindex(K::KernelMatrix, i::Int, j::Int) = K.f(K.X[i], K.Y[j])

rowelements(K::KernelMatrix) = K.X
colelements(K::KernelMatrix) = K.Y
kernel(K::KernelMatrix) = K.f

function KernelMatrix(f, X, Y)
    T = Base.promote_op(f, eltype(X), eltype(Y))
    return KernelMatrix{typeof(f),typeof(X),typeof(Y),T}(f, X, Y)
end

"""
    assembel_hmatrix(K::AbstractKernelMatrix[; atol, rank, rtol, kwargs...])

Construct an approximation of `K` as an [`HMatrix`](@ref) using the partial ACA
algorithm for the low rank blocks. The `atol`, `rank`, and `rtol` optional
arguments are passed to the [`PartialACA`](@ref) constructor, and the remaining
keyword arguments are forwarded to the main `assemble_hmatrix` function.
"""
function assemble_hmatrix(
    K::AbstractKernelMatrix;
    atol = 0,
    rank = typemax(Int),
    rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
    kwargs...,
)
    comp = PartialACA(; rtol, atol, rank)
    adm = StrongAdmissibilityStd()
    X = map(center, rowelements(K))
    Y = map(center, colelements(K))
    Xclt = ClusterTree(X)
    Yclt = ClusterTree(Y)
    return assemble_hmatrix(K, Xclt, Yclt; adm, comp, kwargs...)
end
