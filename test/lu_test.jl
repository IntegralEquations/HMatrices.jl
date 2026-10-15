using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

using HMatrices: RkMatrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

Random.seed!(1)

m = 5000
T = Float64
X = points_on_sphere(m)
Y = X

K = laplace_matrix(X, X)

splitter = CardinalitySplitter(; nmax = 50)
Xclt = ClusterTree(X, splitter)
Yclt = ClusterTree(Y, splitter)
adm = StrongAdmissibilityStd(3)
comp = PartialACA(; atol = 1.0e-10)
for threads in (false, true)
    H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads, distributed = false)
    hlu = lu(H; atol = 1.0e-10)
    y = rand(m)
    M = Matrix(K)
    exact = M \ y
    approx = hlu \ y
    @test norm(exact - approx, Inf) < 1.0e-10
    # test multiplication by checking if the solution is correct
    @test hlu.L * (hlu.U * approx) ≈ y
    # a compressor with its own buffer type
    hlu = lu(H, ACAWithRecompression(PartialACA(; atol = 1.0e-10)))
    @test norm(exact - hlu \ y, Inf) < 1.0e-10
end

@testset "different row and column trees" begin
    # same points in a different order: the row and column permutations differ.
    # The self term of the kernel is comparable to its neighbours, since the LU
    # factorization does not pivot.
    n = 1500
    X = points_on_sphere(n)
    Y = X[randperm(n)]
    f = let w = 4π / n
        (x, y) -> x == y ? sqrt(w / π) / 2 : w / (4π * norm(x - y))
    end
    K = KernelMatrix(f, X, Y)
    H = assemble_hmatrix(K, ClusterTree(X, splitter), ClusterTree(Y, splitter); adm, comp, threads = false)
    y = rand(n)
    exact = Matrix(K) \ y
    @test norm(lu(H; atol = 1.0e-10) \ y - exact) < 1.0e-6 * norm(exact)
end
