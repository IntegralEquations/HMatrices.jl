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
comp = PartialACA(; atol = 1e-10)
for threads in (false, true)
    H = assemble_hmatrix(Hermitian(K), Xclt, Yclt; adm, comp, threads, distributed = false)
    hchol = cholesky(H; atol = 1e-10)
    y = rand(m)
    M = Matrix(K)
    exact = M \ y
    approx = hchol \ y
    @test norm(exact - approx, Inf) < 1e-10
    # test multiplication by checking if the solution is correct
    @test hchol.L * (hchol.U * approx) ≈ y
end
