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
X = rand(Geometry.Point2D,100)
Y = rand(Geometry.Point2D,100)
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

Base.size(K::KernelMatrix)                   = length(K.X), length(K.Y)
Base.getindex(K::KernelMatrix,i::Int,j::Int) =  K.f(K.X[i],K.Y[j])

rowelements(K::KernelMatrix) = K.X
colelements(K::KernelMatrix) = K.Y
kernel(K::KernelMatrix) = K.f

function KernelMatrix(f,X,Y)
    T = Base.promote_op(f,eltype(X),eltype(Y))
    KernelMatrix{typeof(f),typeof(X),typeof(Y),T}(f,X,Y)
end
