# [Kernel matrices](@id kernelmatrix-section)

While in the [introduction](@ref assemble-generic-subsection) we presented a
somewhat general way to assemble an `HMatrix`, abstract matrices associated with
an underlying kernel function are common enough in boundary integral equations
that a special interface exists for facilitating their use.

The [`AbstractKernelMatrix`](@ref) interface is used to represent matrices `K` with
`i,j` entry given by `f(X[i],Y[j])`, where `X=rowelements(K)` and
`Y=colelements(K)`. The row and columns elements may be as simple as points in
$\mathbb{R}^d$ (as is the case for Nyström methods), but they can also be more
complex objects such as triangles or basis functions --- the only thing required is
that `f(X[i],Y[j])` make sense. 

A concrete implementation of `AbstractKernelMatrix` is provided by the
`KernelMatrix` type. Creating the matrix associated with the Helmholtz
free-space Greens function, for example, can be accomplished through:

```@example kernel-matrix
using HMatrices, LinearAlgebra, StaticArrays
const Point3D = SVector{3,Float64}
X = rand(Point3D,10_000)
Y = rand(Point3D,10_000)
const k = 2π
function G(x,y) 
    d = norm(x-y)
    exp(im*k*d)/(4π*d)
end
K = KernelMatrix(G,X,Y)
```

Compressing `K` is now as simple as:

```@example kernel-matrix
H = assemble_hmatrix(K;rtol=1e-6)
```

It is worth noting that several *default* choices are made during the
compression above. See the [Assembling and `HMatrix`](@ref
assemble-generic-subsection) section or the documentation of
[`assemble_hmatrix`](@ref) for information on how to obtain a more granular control
of the assembling stage.

As before, you can multiply `H` by a vector, or do an `lu` factorization of it.

## Support for tensor kernels

!!! warning
    Support for tensor-valued kernels should be considered experimental at this
    stage.

For vector-valued partial differential equations such as *Stokes* or
time-harmonic *Maxwell's* equation, the underlying integral operator has a
kernel function which is a tensor. This package currently provides some limited
support for these types of operators. The example below illustrates how to build
an [`HMatrix`](@ref) representing a [`KernelMatrix`](@ref) corresponding to Stokes Greens function for points on a sphere:

```@example stokes
using HMatrices, LinearAlgebra, StaticArrays
const Point3D = SVector{3,Float64}
m = 5_000
X = Y = [Point3D(sin(θ)cos(ϕ),sin(θ)*sin(ϕ),cos(θ)) for (θ,ϕ) in zip(π*rand(m),2π*rand(m))]
const μ = 5
function G(x,y)
    r = x-y
    d = norm(r) + 1e-10
    1/(8π*μ) * (1/d*I + r*transpose(r)/d^3)
end
K = KernelMatrix(G,X,Y)
H = assemble_hmatrix(K;atol=1e-4)
```

You can now multiply `H` by a density `σ`, where `σ` is a `Vector` of
`SVector{3,Float64}`

```@example stokes
σ = rand(SVector{3,Float64},m)
y = H*σ
# test the output agains the exact value for a given `i`
i = 42
y[i] - sum(K[i,j]*σ[j] for j in 1:m)
```

!!! note
    The *naive* idea of reinterpreting these *matrices of tensors* as a (larger)
    matrix of scalars does not always work because care to be taken when
    choosing the pivot in the compression stage of the [`PartialACA`](@ref) in
    order to exploit some analytic properties of the underlying kernel. See
    e.g. section 2.3 of [this
    paper](https://www.sciencedirect.com/science/article/pii/S0021999117306721)
    for a brief discussion.

## Vectorized kernels and local indices

A more efficient implementation of your kernel `K::AbstractKernelMatrix` can
sometimes lead to faster *assembling* times. In particular, providing a permuted
kernel `Kp` using the local indexing system of the `HMatrix` (and setting the
keyword argument `global_index=false` in `assemble_hmatrix`) avoids frequent
unnecessary index permutations, and can facilitate vectorization. This is
because the permuted kernel `Kp` will be called through
`Kp[I::UnitRange,J::UnitRange]` to fill in the dense blocks of the matrix, through
`Kp[I::UnitRange,j::int]` and `adjoint(Kp)[I::UnitRange,j]` to build a low-rank
approximation of compressible blocks. 

The vectorization example in the [Notebook section](@ref notebook-section) shows how a custom (and somewhat
more complex) implementation of a vectorized Laplace kernel using the
[LoopVectorization](https://github.com/JuliaSIMD/LoopVectorization.jl) package
can lead to faster (sequential) execution.
