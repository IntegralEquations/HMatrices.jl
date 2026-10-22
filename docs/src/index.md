```@meta
CurrentModule = HMatrices
```

# [HMatrices.jl](@id home-section)

*A package for assembling and factoring hierarchical matrices*

## Overview

This package provides some functionality for assembling as well as for doing linear
algebra with [hierarchical
matrices](https://en.wikipedia.org/wiki/Hierarchical_matrix). The main structure
exported is the [`HMatrix`](@ref) type, which can be used to efficiently
approximate certain linear operators containing a hierarchical low-rank
structure. Once assembled, a hierarchical matrix can be used to accelerate the
solution of `Ax=b` in a variety of ways. Below you will find a quick
introduction for how to *assemble* and *utilize* an [`HMatrix`](@ref); see the
[References](@ref references-section) section for more information on the
available methods and structures.

!!! note 
    Although hierarchical matrices have a broad range of application, this
    package focuses on their use to approximate integral operators arising in
    **boundary integral equation (BIE)** methods. As such, most of the API has
    been designed with BIEs in mind, and the examples that follow will focus on
    the compression of integral operators. Feel free to open an
    [issue](https://github.com/IntegralEquations/HMatrices.jl/issues/new) or reach out if
    you have an interesting application of hierarchical matrices in mind not
    covered by this package!

!!! tip "Useful references"
    The notation and algorithms implemented were mostly drawn from the following
    references:
    - Hackbusch, Wolfgang. *Hierarchical matrices: algorithms and analysis*. Vol. 49. Heidelberg: Springer, 2015.
    - Bebendorf, Mario. *Hierarchical matrices*. Springer Berlin Heidelberg, 2008.
## [Assembling an `HMatrix`](@id assemble-generic-subsection)

To assemble an [`HMatrix`](@ref) you need two ingredients:

1. the matrix-like object `K` that you wish to compress, which only needs to
   support `K[i,j]`;
2. points associated with the rows and the columns of `K`, which are used to
   group the indices into clusters whose interactions can be compressed.

To illustrate how this is done for a concrete problem, consider two set of
points $X = \left\{ \boldsymbol{x}_i \right\}_{i=1}^m$ and $Y
=\left\{\boldsymbol{x}_j \right\}_{j=1}^n$ in $\mathbb{R}^2$, and let `K` be a
$m \times n$ matrix with entries given by:

```math
  K_{i,j} = G(\boldsymbol{x}_i,\boldsymbol{y}_j)
```

for some kernel function $G$. To make things simple, we will take $X$ and
$Y$ to be points distributed on a circle:

```@example assemble-basic
using Random # hide
Random.seed!(1) # hide
using HMatrices, LinearAlgebra, StaticArrays
const Point2D = SVector{2,Float64}

# points on a circle
m = n = 10_000
X = Y = [Point2D(sin(i*2π/n),cos(i*2π/n)) for i in 0:n-1]
nothing
```

Next we will create the matrix-like structure to represent the object `K`. We
will pick `G` to be the free-space Greens function for Laplace's equation in
two-dimensions:

```@example assemble-basic
struct LaplaceMatrix <: AbstractMatrix{Float64}
  X::Vector{Point2D}
  Y::Vector{Point2D}
end

Base.getindex(K::LaplaceMatrix,i::Int,j::Int) = -1/2π*log(norm(K.X[i] - K.Y[j]) + 1e-10)
Base.size(K::LaplaceMatrix) = length(K.X), length(K.Y)

# create the abstract matrix
K = LaplaceMatrix(X,Y)
```

We can now assemble an approximation of `K`, passing the points associated with its
rows and columns:

```@example assemble-basic
H = assemble_hmatrix(K, X, Y; atol = 1e-6, threads = false)
```

The keyword arguments of [`assemble_hmatrix`](@ref) control the accuracy of the
low-rank blocks (`atol`, `rtol` and `rank`), which blocks are compressed (`eta`), and
the size of the smallest clusters (`nmax`). If `K` is a `Hermitian` matrix, wrapping
it as `Hermitian(K)` assembles only half of it and allows for a Cholesky
factorization. When `K` is a [`KernelMatrix`](@ref), the points can be omitted (see the
[Kernel matrices section](@ref kernelmatrix-section)).

!!! tip "Finer control"
    The assembly can also be given its ingredients directly: cluster trees
    partitioning the rows and the columns, an admissibility condition to decide
    which blocks are compressed, and a compressor for those blocks. For example,
    ```julia
    using HMatrices: ClusterTree, CardinalitySplitter, StrongAdmissibilityStd, PartialACA
    Xclt = ClusterTree(X, CardinalitySplitter(; nmax = 32))
    H = assemble_hmatrix(K, Xclt, Xclt; adm = StrongAdmissibilityStd(; eta = 2), comp = PartialACA(; atol = 1e-6))
    ```

You can now use `H` *in lieu* of `K` (as an approximation) for certain linear
algebra operations, as shown next.

## Matrix vector product and iterative solvers

The simplest operation you can perform with an `HMatrix` is to multiply it by a
vector:
```@example assemble-basic
x = rand(n)
norm(H*x - K*x)
```

More advanced options (such as choosing between a threaded or serial
implementation) can be accessed by calling [`mul!`](@ref)
directly:

```@example assemble-basic
y = similar(x)
mul!(y,H,x,1,0;threads=false)
norm(y - K*x)
```

The example below illustrates how to use the `HMatrix` `H` constructed above
with the
[`IterativeSolvers`](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl)
package:

```@example assemble-basic
using IterativeSolvers
b = rand(m)
approx = gmres!(y,H,b;abstol=1e-6)
exact  = Matrix(K)\b
norm(approx-exact)/norm(exact)
```

Internally, `H` is stored in the ordering induced by the clustering of the points,
but every operation, such as `H*x`, `mul!`, `Matrix(H)`,
or the solves with a factorization, works in the ordering of the points given by the
user. The cost of permuting the vectors is negligible compared to that of the
products.

!!! note "Problem size"
    For "small" problem sizes, the overhead associated with the more complex
    structure of an `HMatrix` will lead to computational times that are larger
    than the *dense* representation, even when the `HMatrix` occupies less
    memory. For large problem sizes, however, the loglinear complexity will yield
    significant gains in terms of memory and cpu time provided the underlying
    operator has a hierarchical low-rank structure.

## Factorization and direct solvers

Although the forward map illustrated in the example above suffices to solve the
linear system `Kx = b` using an iterative solver, there are circumstances where
a *direct* solver is desirable (because, e.g., the system is not
well-conditioned or you wish to solve it for many right-hand-sides `b`). The
**hierarchical LU** factorization of `H` can be computed as follows (and a
hierarchical Cholesky factorization is available for an `H` assembled from a
`Hermitian` matrix):

```@example assemble-basic
F = lu(H;atol=1e-6)
```

Note that unliked the matrix-vector product, factoring `H` is not *exact* in the
sense that `lu(H) ≠ lu(Matrix(H))`. The accuracy of the approximation can be
controlled through the keyword arguments `atol`, `rtol` and `rank`, which are used
in the various intermediate truncations performed during the
factorization. See [`lu`](@ref) for more details.

!!! important "Truncation error"
    The parameters `atol` and `rtol` are used to control the truncation of
    low-rank blocks *adaptively* using an estimate of the true error (in
    Frobenius norm). These local errors may accumulate after successive
    truncations, meaning that the global approximation error (in Frobenius norm) may be larger
    than the prescribed tolerance.

The returned object `F` is of the `LU` type, and efficient
routines are provided to solve linear system using `F`:

```@example assemble-basic
approx = F\b
norm(approx-exact)/norm(exact)
```

Like `H`, `F \ b` works in the ordering of the points. The factors `F.L` and `F.U`
are stored in the ordering of the clusters, and
`F.L * (F.U * x) ≈ Matrix(H)[F.p, F.q] * x`, as for the pivoted factorizations of
`LinearAlgebra`.

Note that the error in solving the linear system may be significantly larger
than the error in computing `H*x` due to the condition of the underlying
operator.

!!! tip
    Because factoring an `HMatrix` with a small error tolerance can be quite
    time-consuming, a hybrid strategy commonly employed consists of using a
    rough factorization (with e.g. large tolerance or a fixed rank) as a
    preconditioner to an iterative solver.

## Other kernels

So far we focused on the (manual) compression of a simple kernel matrix where
the entry `(i,j)` depended only on a function `G` and on point-clouds `X` and
`Y`. There are many interesting applications where computing the `(i,j)` entry
requires more information, such as triangles, basis functions, or normal
vectors. To illustrate how the methods discussed before could be adapted lets
construct a double-layer matrix for Laplace equation. To keep the example
simple, we will re-use the point clouds `X` and `Y` defined before, and we will
simply append the normal vector information to a `LaplaceDoubleLayer` structure.
The implementation could look something like this:

```@example assemble-basic
struct LaplaceDoubleLayer <: AbstractMatrix{Float64}
    X::Vector{Point2D}
    Y::Vector{Point2D}
    NY::Vector{Point2D} # normals at Y coordinate
end

function Base.getindex(K::LaplaceDoubleLayer,i::Int,j::Int) 
    r = K.X[i] - K.Y[j]
    d = norm(r) + 1e-10
    return (1 / (2π) / (d^2) * dot(r, K.NY[j]))
end
Base.size(K::LaplaceDoubleLayer) = length(K.X), length(K.Y)
```

We can now simply instantiate a double-layer kernel, and compress it as before

```@example assemble-basic
# create the abstract matrix
ny = Y
K = LaplaceDoubleLayer(X,Y,ny)
H = assemble_hmatrix(K, X, Y; atol = 1e-6, threads = false)
```

With `H` assembled, everything else works exactly as before!

## Index

```@index
```
