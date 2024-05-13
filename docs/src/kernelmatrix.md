# [Kernel matrices](@id kernelmatrix-section)

While in the [introduction](@ref assemble-generic-subsection) we presented a
somewhat general way to assemble an `HMatrix`, abstract matrices associated with
an underlying kernel function are common enough in boundary integral equations
that a special interface exists for facilitating their use.

The [`AbstractKernelMatrix`](@ref) interface is used to represent matrices `K` with
`i,j` entry given by `f(X[i],Y[j])`, where `X=rowelements(K)` and
`Y=colelements(K)`. The row and columns elements may be as simple as points in
$\mathbb{R}^d$ (as is the case for Nyström methods), but they can also be more
complex objects such as triangles or basis functions. In such cases, it is
required that `center(X[i])` and `center(Y[j])` return a point as an `SVector`,
and that `f(X[i],Y[j])` make sense.

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
compression above. See the [Assembling an `HMatrix`](@ref
assemble-generic-subsection) section or the documentation of
[`assemble_hmatrix`](@ref) for information on how to obtain a more granular
control of the assembling stage.

As before, you can multiply `H` by a vector, or do an `lu` factorization of it.

Finally, here is a somewhat contrived example of how to use a `KernelMatrix`
when the `rowelements` and `colelements` are not simply points (as required e.g.
in a Galerkin discretization of boundary integral equations):

```@example galerkin-kernel
using HMatrices, LinearAlgebra, StaticArrays
# create a simple structure to represent a segment
struct Segment
    start::SVector{2,Float64}
    stop::SVector{2,Float64}
end
# extend the function center to work with segments
HMatrices.center(s::Segment) = 0.5*(s.start + s.stop)
# P1 mesh of a circle
npts = 10_000
nodes = [SVector(cos(s), sin(s)) for s in range(0,stop=2π,length=npts+1)]
segments = [Segment(nodes[i],nodes[i+1]) for i in 1:npts]
# Now define a kernel function that takes two segments (instead of two points) and returns a scalar
function G(target::Segment,source::Segment) 
    x, y = HMatrices.center(target), HMatrices.center(source)
    d = norm(x-y)
    return -log(d + 1e-10)
end
K = KernelMatrix(G,segments,segments)
# compress the kernel matrix
H = assemble_hmatrix(K;rtol=1e-6)
```

## Support for tensor kernels

!!! warning
    Support for tensor-valued kernels should be considered experimental at this
    stage.

For vector-valued partial differential equations such as *Stokes* or
time-harmonic *Maxwell's* equation, the underlying integral operator has a
kernel function which is a tensor. This package currently provides some limited
support for these types of operators. The example below illustrates how to build
an [`HMatrix`](@ref) representing a [`KernelMatrix`](@ref) corresponding to
Stokes Greens function for points on a sphere:

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
exact = sum(K[i,j]*σ[j] for j in 1:m)
@assert norm(y[i] - exact) < 1e-4 # hide
@show y[i] - exact
```

!!! note
    The *naive* idea of reinterpreting these *matrices of tensors* as a (larger)
    matrix of scalars does not always work because care to be taken when
    choosing the pivot in the compression stage of the [`PartialACA`](@ref) in
    order to exploit some analytic properties of the underlying kernel. See
    e.g. section 2.3 of [this
    paper](https://www.sciencedirect.com/science/article/pii/S0021999117306721)
    for a brief discussion.
