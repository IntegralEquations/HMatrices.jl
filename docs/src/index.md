```@meta
CurrentModule = HMatrices
```

# HMatrices.jl

*A package for assembling and factoring hierarchical matrices with a focus on
boundary integral equations*

This package provides various methods for assembling as well as for doing linear
algebra with [hierarchical
matrices](https://en.wikipedia.org/wiki/Hierarchical_matrix). The main structure
exported is the [`HMatrix`](@ref) type, which can be used to efficiently
approximate certain linear operators containing a hierarchical low-rank
structure. Once assembled, a hierarchical matrix can be used to accelerate the
solution of `Ax=b` in a variety of ways, as illustrated in the examples that follow.

!!! note
    Although hierarchical matrices have a broad range of application, this
    package focuses on their use to approximate integral
    operators arising in **boundary integral equation methods**. 

## Assembling an `HMatrix`

In order to assemble an [`HMatrix`](@ref), you need the following
(problem-specific) ingredients:

1. The matrix-like object `K` that you wish to compress
2. A hierarchical partition of the rows and columns indices of `K`
3. An admissibility condition for determining (*a priory*) whether a block is
   compressible, and
4. A function/functor to generate a low-rank approximation of compressible
   blocks

To illustrate how this is done for a concrete problem, consider two set of
points $X = \left\{ \boldsymbol{x}_i \right\}_{i=1}^m$ and $Y = \left\{
\boldsymbol{x}_j \right\}_{j=1}^m$ in $\mathbb{R}^3$, and let `A`
be an $m \times n$ matrix with entries given by:

```math
  A_{i,j} = \exp(-||\boldsymbol{x}_i-\boldsymbol{y}_j||)
```

The following code will create the point clouds `X` and `Y`, as well as the matrix-like object `K`:

```@example assemble-basic
using HMatrices, LinearAlgebra, StaticArrays
const Point3D = SVector{3,Float64}

struct ExponentialMatrix <: AbstractMatrix{Float64}
  X::Vector{Point3D}  
  Y::Vector{Point3D}
end

Base.getindex(K::ExponentialMatrix,i::Int,j::Int) = exp(-norm(K.X[i] - K.Y[j]))
Base.size(K::ExponentialMatrix) = length(K.X), length(K.Y)

# points on a sphere
m = n = 10000
X = Y = [Point3D(sin(θ)cos(ϕ),sin(θ)*sin(ϕ),cos(θ)) for (θ,ϕ) in zip(π*rand(n),2π*rand(n))]
# create the abstract matrix
K = ExponentialMatrix(X,Y)
```

The next step in to partition the point clouds `X` and `Y` into a tree-like data
structure so that blocks corresponding to well-separated points can be easily
distinguished and compressed. The `WavePropBase` package provides the
`ClusterTree` struct for this purpose:

```@example assemble-basic
Xclt = Yclt = ClusterTree(X)
nothing # hide
```

The third requirement is an *admissibilty condition* to determine if the
interaction between two clusters should be compressed. We will use the
[`StrongAdmissibilityStd`](@ref):

```@example assemble-basic
adm = StrongAdmissibilityStd()
nothing # hide
```

The final step is to provide a method to compress admissible blocks. Here we
will use the [`PartialACA`](@ref) functor:

```@example assemble-basic
comp = PartialACA(;atol=1e-6)
nothing # hide
```

We can now compress the operator `K` using

```@example assemble-basic
H = HMatrix(K,Xclt,Yclt,adm,comp;threads=false,distributed=false)
```

## Matrix vector product and iterative solvers

You can now use `H` as an approximation to `K`:

```@example assemble-basic
x = rand(n)
H*x ≈ K*x 
```

The larger the number of points `n`, the more we expect to gain from this
compression technique.

## Factorization and direct solvers

Although the forward map illustrated in the example above suffices to solve the
linear system `Kx = b`, there are circumnstances when a *direct* solver is more
appropriate (because, e.g., the system is not well-conditioned or you wish to
solve it for many right-hand-sides `b`). At present, you can do an `lu`
factorization of `H` as follows:

```@example assemble-basic
F = lu(H,comp)
```

The factorization `F` can then be used to solve a linear system:

```@example assemble-basic
b = rand(m)
approx = F\b
exact  = Matrix(K)\b
norm(approx-exact,Inf)
```

!!! note
    Most algebraic manipulations of hierarchical matrices require the use of
    *compressor* to keep the rank of the low blocks under control when e.g.
    adding two low-rank blocks. In `HMatrices.jl`, there are a few options
    defined in
    [`src/compressor.jl`](https://github.com/WaveProp/HMatrices.jl/blob/main/src/compressor.jl).


Note that the error in solving the linear system may be significantly larger
than the error in computing `H*x` due to the condition of the underlying
operator.

!!! tip
    Because factoring an `HMatrix` with a small error tolerance can be quite
    time-consuming, a hybrid strategy consists of using a rough factorization
    (with a large tolerance) as a preconditioner to an iterative solver.

## Tensor-valued matrices

Some support is currently provided for matrices with entries which are
themselves matrices...
## References

[1] Hackbusch, Wolfgang. Hierarchical matrices: algorithms and analysis. Vol. 49. Heidelberg: Springer, 2015.

[2] Bebendorf, Mario. Hierarchical matrices. Springer Berlin Heidelberg, 2008.

```@index
```

```@autodocs
Modules = [HMatrices]
```
