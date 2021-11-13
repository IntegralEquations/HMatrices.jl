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
    [issue](https://github.com/WaveProp/HMatrices.jl/issues/new) or reach out if
    you have an interesting application of hierarchical matrices in mind not
    covered by this package!

!!! tip "Useful references"
    The notation and algorithms implemented were mostly drawn from the following
    references:
    - Hackbusch, Wolfgang. *Hierarchical matrices: algorithms and analysis*. Vol. 49. Heidelberg: Springer, 2015.
    - Bebendorf, Mario. *Hierarchical matrices*. Springer Berlin Heidelberg,
    2008.
    
## [Assembling an `HMatrix`](@id assemble-generic-subsection)

In order to assemble an [`HMatrix`](@ref), you need the following
(problem-specific) ingredients:

1. The matrix-like object `K` that you wish to compress
2. A `rowtree` and `coltree` providing a hierarchical partition of the rows and columns of `K`
3. An admissibility condition for determining (*a priory*) whether a block given
   by a node in the `rowtree` and node in the `coltree` is compressible
4. A function/functor to generate a low-rank approximation of compressible
   blocks

To illustrate how this is done for a concrete problem, consider two set of
points $X = \left\{ \boldsymbol{x}_i \right\}_{i=1}^m$ and $Y
=\left\{\boldsymbol{x}_j \right\}_{j=1}^n$ in $\mathbb{R}^3$, and let `K` be a 
$m \times n$ matrix with entries given by:

```math
  K_{i,j} = G(\boldsymbol{x}_i,\boldsymbol{y}_j)
```

for some kernel function $G$. To make things simple, we will take $X$ and
$Y$ to be points distributed on a circle:

```@example assemble-basic
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

The next step consists in partitioning the point clouds `X` and `Y` into a
tree-like data structure so that blocks corresponding to well-separated points
can be easily distinguished and compressed. The `WavePropBase` package provides
the `ClusterTree` struct for this purpose (see its documentation for more
details on available options):

```@example assemble-basic
Xclt = Yclt = ClusterTree(X)
nothing # hide
```

The object `Xclt` represents a tree partition of the point cloud into
axis-aligned bounding boxes.

The third requirement is an *admissibilty condition* to determine if the
interaction between two clusters should be compressed. We will use the
[`StrongAdmissibilityStd`](@ref), which is appropriate for *asymptotically
smooth kernels* such as the one considered:

```@example assemble-basic
adm = StrongAdmissibilityStd()
nothing # hide
```

The final step is to provide a method to compress admissible blocks. Here we
will use the [`PartialACA`](@ref) functor implementing an *adaptive
cross approximation* with partial pivoting strategy:

```@example assemble-basic
comp = PartialACA(;atol=1e-6)
nothing # hide
```

With these ingredients at hand, we can assemble an approximation for `K` using

```@example assemble-basic
H = assemble_hmat(K,Xclt,Yclt;adm,comp,threads=false,distributed=false)
```

!!! important
    The [`assemble_hmat`](@ref) function is the main constructor exported by this
    package, so it is worth getting familiar with it and the various keyword
    arguments it accepts.

!!! note
    Reasonable defaults exist for the *admissibility condition*, *cluster tree*,
    and *compressor* when the kernel `K` is an [`AbstractKernelMatrix`](@ref),
    so that the construction process is somewhat simpler than just presented in
    those cases. Manually constructing each ingredient, however, gives a level
    of control not available through the default constructors. See the [Kernel
    matrices section](@ref kernelmatrix-section) for more details.

You can now use `H` *in lieu* of `K` (as an approximation) for certain linear
algebra operations, as shown next.

!!! tip "Disabling `getindex`"
    Although the `getindex(H,i::Int,j::Int)` method is defined for an `AbstractHMatrix`, its use
    is mostly for display purposes in a `REPL` environment, and should be
    avoided in any linear algebra routine. To avoid the performance pitfalls
    related to methods falling back to the generic `LinearAlgebra`
    implementation of various algorithms (which make use `getindex`
    extensively), you can disable `getindex` on types defined in this package by
    calling [`HMatrices.disable_getindex`](@ref). The consequence is that
    calling `getindex(H,i,j)` will throw an error. If a given operation is
    running unexpectedly slow, try disabling `getindex` to see if any part of
    the code is falling back to a generic implementation.

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
mul!(y,H,x,1,0;threads=false,global_index=true)
norm(H*x - K*x)
```

!!! important "Local and global indices"
    The hierarchical matrix `H` above is stored as `H = Pr*_H*Pc`, where `Pr`
    and `Pc` are row and column permutation matrices, respectively. It is
    sometimes convenient to work directly with `_H` for performance reasons (for
    example, in an iterative solver, you may want to permute rows and columns
    only once *offline* and perform the matrix multiplication with `_H`); the
    keyword argument `global_index=false` can be passed to perform the desired
    operations on `_H` instead.

!!! note "Problem size"
    For "small" problem sizes, the overhead associated to the more complex
    structure of the `HMatrix` will lead to computational times that are larger
    than the *dense* representation, even when the `HMatrix` occupies less
    memory. For large problem sizes, however, the linear complexity will yield
    significant gains in terms of memory and cpu time provided the underlying
    operator has a hierarchical low-rank structure.

## Factorization and direct solvers

Although the forward map illustrated in the example above suffices to solve the
linear system `Kx = b` using an iterative solver, there are circumstances where
a *direct* solver is desirable (because, e.g., the system is not
well-conditioned or you wish to solve it for many right-hand-sides `b`). At
present, the only available factorization is the **hierarchical lu** factorization of `H`,
which can be accomplished as follows:

```@example assemble-basic
F = lu(H;atol=1e-6)
```

Note that unliked the matrix-vector product, factoring `H` is not *exact* in the
sense that `lu(H) ≠ lu(Matrix(H))`. The accuracy of the approximation can be
controlled through the keyword arguments `atol,rol` and `rank`, which are used
in the various intermediate truncations performed during the
factorization. See [`lu`](@ref) for more details.

!!! important "Truncation error"
    The parameters `atol` and `rtol` are used to control the truncation of
    low-rank blocks *adaptively* using an estimate of the true error (in
    Frobenius norm). These local error may accumulate after successive
    truncations, which in practice means that the final errors can be somewhat
    larger than the prescribed tolerance. Because of this, you may want to set
    `atol` and `rtol` to be somewhat smaller than your target accuracy.

The returned object `F` is of the `LU` type, and efficient
routines are provided to solve linear system using `F`:

```@example assemble-basic
b = rand(m)
approx = F\b
exact  = Matrix(K)\b
norm(approx-exact)/norm(exact)
```

Note that the error in solving the linear system may be significantly larger
than the error in computing `H*x` due to the condition of the underlying
operator.

!!! tip
    Because factoring an `HMatrix` with a small error tolerance can be quite
    time-consuming, a hybrid strategy commonly employed consists of using a
    rough factorization (with e.g. large tolerance or a fixed rank) as a
    preconditioner to an iterative solver.

## Index

```@index
```