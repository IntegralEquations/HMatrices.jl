using Distributed
addprocs(4; exeflags = `--project=$(Base.active_project())`)

using Test
using StaticArrays
@everywhere using HMatrices
@everywhere include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

@testset "Assemble" begin
    m, n = 10_000, 10_000
    X = Y = points_on_cylinder(m, 1)
    splitter = CardinalitySplitter(; nmax = 100)
    Xclt = Yclt = ClusterTree(X, splitter)
    adm = StrongAdmissibilityStd(; eta = 3)
    atol = 1.0e-6
    comp = PartialACA(; atol)
    # Laplace
    K = laplace_matrix(X, Y)
    H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, distributed = false, threads = true)
    Hd = assemble_hmatrix(K, Xclt, Yclt; adm, comp, distributed = true, threads = false)
    x = rand(n)
    y = rand(m)
    @test H * x ≈ Hd * x
end
