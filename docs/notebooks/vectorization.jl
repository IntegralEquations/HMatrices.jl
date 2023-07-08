### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 2b4aa264-4145-11ec-0d24-ab437ba0f84e
begin
    using Pkg: Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())

    using HMatrices, Plots, PlutoUI, LinearAlgebra, StaticArrays, WavePropBase,
          BenchmarkTools, LoopVectorization
    using WavePropBase: PlotPoints, PlotTree, root_elements
end

# ╔═╡ 9efa3b28-900c-4515-8aba-617bce2ddc68
PlutoUI.TableOfContents(; depth=2)

# ╔═╡ d1917856-64d1-4dab-9e05-40b1152217d2
begin
    const Point3D = SVector{3,Float64}
    m = 10_000
    X = Y = [Point3D(sin(θ)cos(ϕ), sin(θ) * sin(ϕ), cos(θ))
             for (θ, ϕ) in zip(π * rand(m), 2π * rand(m))]
end

# ╔═╡ bdfebc7a-a7a6-4fa6-8884-b4f86dc55cb3
begin
    Xclt = Yclt = ClusterTree(X)
    plot(PlotTree(), Xclt; mc=:red)
end

# ╔═╡ 306d493b-a558-4e1d-bad6-5c295da79e82
begin
    function G(x, y)
        d = norm(x - y) + 1e-8
        return inv(4π * d)
    end
    K = KernelMatrix(G, X, Y)
end

# ╔═╡ 99041c6a-c12f-4a09-8bfd-cf769fd258a6
H = assemble_hmat(K, Xclt, Yclt; threads=false)

# ╔═╡ 02d89923-e478-4778-8b80-eaa3357a6a8c
b = @benchmark assemble_hmat($K, $Xclt, $Yclt; threads=false)

# ╔═╡ fca099bf-25c0-4ee8-b3f0-c042c4e5fde3
begin
    Xp = Yp = root_elements(Xclt)
    Kp = KernelMatrix(G, Xp, Yp)
    Hp = assemble_hmat(Kp, Xclt, Yclt; threads=false, global_index=false)
end

# ╔═╡ 37ed939e-1738-4ade-85d7-526c3c4ab536
bp = @benchmark assemble_hmat($Kp, Xclt, Yclt; threads=false, global_index=false)

# ╔═╡ c6729646-0b6c-4ea2-b692-62e1e8f6b97c
begin
    struct LaplaceMatrixVec <: AbstractKernelMatrix{Float64}
        X::Matrix{Float64}
        Y::Matrix{Float64}
    end
    Base.size(K::LaplaceMatrixVec) = size(K.X, 1), size(K.Y, 1)

    # convenience constructor based on Vector of StaticVector
    function LaplaceMatrixVec(_X::Vector{Point3D}, _Y::Vector{Point3D})
        X = collect(transpose(reshape(reinterpret(Float64, _X), 3, :)))
        Y = collect(transpose(reshape(reinterpret(Float64, _Y), 3, :)))
        return LaplaceMatrixVec(X, Y)
    end

    function Base.getindex(K::LaplaceMatrixVec, i::Int, j::Int)
        d2 = (K.X[i, 1] - K.Y[j, 1])^2 + (K.X[i, 2] - K.Y[j, 2])^2 +
             (K.X[i, 3] - K.Y[j, 3])^2
        d = sqrt(d2) + 1e-8
        return inv(4π * d)
    end
    function Base.getindex(K::LaplaceMatrixVec, I::UnitRange, J::UnitRange)
        T = eltype(K)
        m = length(I)
        n = length(J)
        Xv = view(K.X, I, :)
        Yv = view(K.Y, J, :)
        out = Matrix{T}(undef, m, n)
        @turbo for j in 1:n
            for i in 1:m
                d2 = (Xv[i, 1] - Yv[j, 1])^2
                d2 += (Xv[i, 2] - Yv[j, 2])^2
                d2 += (Xv[i, 3] - Yv[j, 3])^2
                d = sqrt(d2) + 1e-8
                out[i, j] = inv(4 * π * d)
            end
        end
        return out
    end
    function Base.getindex(K::LaplaceMatrixVec, I::UnitRange, j::Int)
        T = eltype(K)
        m = length(I)
        Xv = view(K.X, I, :)
        out = Vector{T}(undef, m)
        @turbo for i in 1:m
            d2 = (Xv[i, 1] - K.Y[j, 1])^2
            d2 += (Xv[i, 2] - K.Y[j, 2])^2
            d2 += (Xv[i, 3] - K.Y[j, 3])^2
            d = sqrt(d2) + 1e-8
            out[i] = inv(4 * π * d)
        end
        return out
    end
    function Base.getindex(adjK::Adjoint{<:Any,<:LaplaceMatrixVec}, I::UnitRange, j::Int)
        K = parent(adjK)
        T = eltype(K)
        m = length(I)
        Yv = view(K.Y, I, :)
        out = Vector{T}(undef, m)
        @turbo for i in 1:m
            d2 = (Yv[i, 1] - K.X[j, 1])^2
            d2 += (Yv[i, 2] - K.X[j, 2])^2
            d2 += (Yv[i, 3] - K.X[j, 3])^2
            d = sqrt(d2) + 1e-8
            out[i] = inv(4 * π * d)
        end
        return out
    end
end

# ╔═╡ 85de3038-88f8-442e-a2cd-5462f9552d12
md"""
# Custom kernels and vectorization

In this notebook we will cover how to implement your own `AbstractKernelMatrix` and explore ways to make the `HMatrix` assemble stage faster by exploiting vectorization.  We will focus on computing an approximation to the matrix $K$ with entries given in terms of a Greens function $G$:

```math
K_{i,j} = G(\boldsymbol{x}_i,\boldsymbol{y}_j), \quad \mbox{for} \quad \boldsymbol{x}_i \in X, \quad \boldsymbol{y}_j \in Y,
```
where $X=\{\boldsymbol{x}_k\}_{k=1}^m$ and $Y=\{\boldsymbol{y}_k\}_{k=1}^n$ are sets of points in $\mathbb{R}^3$ and $G(\boldsymbol{x},\boldsymbol{y}) = \frac{1}{4\pi ||\boldsymbol{x} - \boldsymbol{y} ||}$ is the Laplace's freespace Greens function.

After defining the point clouds $X$ and $Y$ (and the required tree data structures to cluster them) we will implement both a simple as well as a more complex (vectorized) implementation of this kernel matrix. 

!!! note "Singular kernels"
	To avoid dealing with singularity at $\boldsymbol{x} = \boldsymbol{y}$, we will 		 simply add a small quantity `EPS = 1e-8`  to the distance. In an actual boundary integral equation solver, some special treatment is usually required to handle entries with $\boldsymbol{x} \approx \boldsymbol{y}$.
"""

# ╔═╡ 22b7faf3-b7e5-4c17-89eb-9f5118517188
md"""
## Point clouds and `ClusterTree`s

The first step is to create the point clouds $X$ and $Y$. We will take $X=Y$, and chose them to be points on a sphere of radius 1:
"""

# ╔═╡ dde4586f-d341-4bf8-a736-50ac311d7c73
md"""
Next we create a `ClusterTree` to organize the points in `X` into a hierarchy of axis-aligned bounding boxes:
"""

# ╔═╡ b29dc077-67ad-4e18-a428-5e832c9f188a
md"""
We can now proceed to the creation of a simple version of the matrix $K$.
"""

# ╔═╡ 78684c89-e7c5-46a0-949f-70e7166e14ef
md"""
## Simple implementation

The first implementation is made as simple as possible. We will make use of the `KernelMatrix` parametric type for this, so we only really need to provide an implementation of $G$:
"""

# ╔═╡ b488948d-dfcd-47ce-8b51-e04048e4eca7
md"""
With `K` at hand, we can now compress it using
"""

# ╔═╡ a513685e-17d3-4252-996b-a7e313561702
md"""
We benchmark the performance for comparison later:
"""

# ╔═╡ 4bfa0d77-58eb-4302-b5d1-05f35c5a500f
md"""
Next we will try to improve on the timing above by implementing a more efficient kernel matrix `K`.
"""

# ╔═╡ 69e228be-cea3-4379-9f9f-42b53abbe810
md"""
## Permuted kernel

Because the local indexing used internally to compute the `HMatrix` differs from the external (global) indexing used, the `getindex` method will not be called on contiguous entries of `K`. Providing a permuted kernel `Kp` may therefore improve performance since call to `Kp` will be made on `UnitRange` indices. This can be accomplished simply by defining `Kp` interms of the (permuted) elements in the `ClusterTree`, and passing the keyword argument `global_index=false` to the constructor to idicate that the kernel passed uses the local indexing system. The code below does precisely this:
"""

# ╔═╡ cd49f749-f0c9-4381-9d2b-03bdafc57584
md"""
Note that the structure of `Hp` is identical to that of `H`. Benchmarking this strategy yields some improvements already:
"""

# ╔═╡ 594751f2-b247-4e10-a43d-16242cc7bc97
md"""
## Vectorized implementation

We will now try to exploit the fact that, during the assemble of the `HMatrix` object, the kernel matrix `K` is called frequently to compute the dense and sparse blocks of the hierarchical matrix.

Dense blocks will be filled by performing the call `Kp[I′,J′]`, where `I′` and `J′` correspond to permutations of unit ranges `I::UnitRange` and `J::UnitRange`. The permutations are present because the indexing system used locally by the `HMatrix` is the one induced by its `rowtree` and `coltree`, which itselfs reorder the elements in the point clouds `X` and `Y` so that they are contiguous within a bounding box (see the documentation of `ClusterTree` and `HMatrix` for more details). Similarly, the low-rank blocks of the `HMatrix` will be assembled by sampling columns of `K` and of its `adjoint` through the following methods: `K[I′,j]` and `adjoint(K)[I′,j::Int]`. We will therefore implement efficient (vectorized) version of these `getindex` methods.

First, let us create a vecotrized kernel matrix withe the required `getindex` methods:
"""

# ╔═╡ 93d306cc-5901-4b0d-91fc-8eddecedad11
md"""
The `LaplaceMatrix` struct above has a vectorized implementatio of `getindex` for `UnitRange` objects. To make sure the `LaplaceMatrix` object is called on `UnitRange` indices, we must therefore constructed using the permuted element `Xp` and `Yp` which use the local index of the `HMatrix`. This is accomplished as follows:
"""

# ╔═╡ 71208d87-a9ed-410e-bad1-f770fc84d4db
begin
    Kv = LaplaceMatrixVec(Xp, Yp)
    Hv = assemble_hmat(Kv, Xclt, Yclt; global_index=false, threads=false)
end

# ╔═╡ 6eaa63e3-755d-482a-a113-98312b6f6fb1
md"""
Benchmarking the new implementation yields:
"""

# ╔═╡ 59fda22e-0547-4845-9207-31c7b526d2f6
bv = @benchmark assemble_hmat($Kv, Xclt, Yclt; threads=false, global_index=false)

# ╔═╡ 0b967aaf-271c-4153-acd6-2e5acff8dc75
md"""
Comparing the times we see that the vectorized implementation is the fastest, and the version with the permuted kernel is faster than the unpermuted one. But these results will vary depending on the machine, number of threads used, and many other factors, so that drawing a general conclusion regarding the speedup of vectorization is difficult. We expect, however, that the vectorized version should be no slower than the non-vectorized one.
"""

# ╔═╡ afc737f1-3523-413f-a73a-96a909580047
minimum(b), minimum(bp), minimum(bv)

# ╔═╡ 44f08eda-b5d3-40d0-aa8b-3cae9e137854
md"""
Finally, not that as expected all of these yield the same matrix, as indicated by the test below:
"""

# ╔═╡ 5fd9b390-ea5f-425c-a624-0cd4a146a897
begin
    x = rand(m)
    y = H * x
    yv = Hv * x
    yp = Hp * x
    er = max(norm(y - yv) / norm(y), norm(y - yp) / norm(y))
    md"""
    Maximum error: $er
    """
end

# ╔═╡ Cell order:
# ╠═2b4aa264-4145-11ec-0d24-ab437ba0f84e
# ╠═9efa3b28-900c-4515-8aba-617bce2ddc68
# ╟─85de3038-88f8-442e-a2cd-5462f9552d12
# ╟─22b7faf3-b7e5-4c17-89eb-9f5118517188
# ╠═d1917856-64d1-4dab-9e05-40b1152217d2
# ╟─dde4586f-d341-4bf8-a736-50ac311d7c73
# ╠═bdfebc7a-a7a6-4fa6-8884-b4f86dc55cb3
# ╟─b29dc077-67ad-4e18-a428-5e832c9f188a
# ╟─78684c89-e7c5-46a0-949f-70e7166e14ef
# ╟─306d493b-a558-4e1d-bad6-5c295da79e82
# ╟─b488948d-dfcd-47ce-8b51-e04048e4eca7
# ╠═99041c6a-c12f-4a09-8bfd-cf769fd258a6
# ╟─a513685e-17d3-4252-996b-a7e313561702
# ╠═02d89923-e478-4778-8b80-eaa3357a6a8c
# ╟─4bfa0d77-58eb-4302-b5d1-05f35c5a500f
# ╟─69e228be-cea3-4379-9f9f-42b53abbe810
# ╠═fca099bf-25c0-4ee8-b3f0-c042c4e5fde3
# ╟─cd49f749-f0c9-4381-9d2b-03bdafc57584
# ╠═37ed939e-1738-4ade-85d7-526c3c4ab536
# ╟─594751f2-b247-4e10-a43d-16242cc7bc97
# ╠═c6729646-0b6c-4ea2-b692-62e1e8f6b97c
# ╟─93d306cc-5901-4b0d-91fc-8eddecedad11
# ╠═71208d87-a9ed-410e-bad1-f770fc84d4db
# ╟─6eaa63e3-755d-482a-a113-98312b6f6fb1
# ╠═59fda22e-0547-4845-9207-31c7b526d2f6
# ╟─0b967aaf-271c-4153-acd6-2e5acff8dc75
# ╠═afc737f1-3523-413f-a73a-96a909580047
# ╟─44f08eda-b5d3-40d0-aa8b-3cae9e137854
# ╟─5fd9b390-ea5f-425c-a624-0cd4a146a897
