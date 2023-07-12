# [Distributed hierarchical matrix](@id dhmatrix-section)

!!! warning
    This is still an experimental feature!

When calling [`assemble_hmatrix`](@ref), the keyword argument `distributed` can be
set to `true` in order to generate a `DHMatrix` object. The main
difference between an `HMatrix` and a `DHMatrix` is that the leaves of a
`DHMatrix` represent a remote reference to an `HMatrix` possibly stored on a
different worker. 

In order to use the distributed capabilities, you must first add the
`Distributed` package and add some workers:

```julia
using Distributed
addprocs(4)
```

You can then load the `HMatrices` package everywhere, and proceed as before:

```julia
@everywhere using HMatrices
using LinearAlgebra, StaticArrays
const Point3D = SVector{3,Float64}
m = 10_000
X = Y = [Point3D(sin(θ)cos(ϕ),sin(θ)*sin(ϕ),cos(θ)) for (θ,ϕ) in zip(π*rand(m),2π*rand(m))]
const μ = 5
function G(x,y)
    r = x-y
    d = norm(r) + 1e-10
    1/(8π*μ) * (1/d*I + r*transpose(r)/d^3)
end
K = KernelMatrix(G,X,Y)
H = assemble_hmatrix(K;atol=1e-4,distributed=true,threads=false)
```

**TODO**: add an interactive notebook example 