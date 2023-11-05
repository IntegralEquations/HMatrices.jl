using Test
using StaticArrays
using HMatrices
using LinearAlgebra

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

@testset "Assemble" begin
    m, n = 1_000, 1_000
    X = Y = rand(SVector{3,Float64}, m)
    splitter = CardinalitySplitter(; nmax = 20)
    Xclt = Yclt = ClusterTree(X, splitter)
    adm = StrongAdmissibilityStd(; eta = 3)
    rtol = 1e-5
    comp = PartialACA(; rtol = rtol)
    # Laplace
    for threads in (true, false)
        K = laplace_matrix(X, Y)
        H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads)
        @test norm(Matrix(K) - Matrix(H; global_index = true)) < rtol * norm(Matrix(K))
        H = assemble_hmatrix(K; threads, distributed = false)
        @test norm(Matrix(K) - Matrix(H; global_index = true)) < rtol * norm(Matrix(K))
        adjH = adjoint(H)
        H_full = Matrix(H; global_index = true)
        adjH_full = adjoint(H_full)

        @testset "getcol" begin
            Hloc = Matrix(H; global_index = false)
            @test Hloc[:, 666] ≈ HMatrices.getcol(H, 666)
            @test adjoint(Hloc)[:, 666] ≈ HMatrices.getcol(adjH, 666)
        end
        # Elastostatic
        K = elastostatic_matrix(X, Y, 1.1, 1.2)
        H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads)
        @test norm(Matrix(K) - Matrix(H; global_index = true)) < rtol * norm(Matrix(K))
    end
end
