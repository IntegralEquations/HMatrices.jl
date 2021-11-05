using Test
using StaticArrays
using HMatrices
using LinearAlgebra

dir = @__DIR__
include(joinpath(dir,"kernelmatrix.jl"))

@testset "Assemble" begin
    m,n  = 1000,1000
    X  = Y   = rand(SVector{3,Float64},m)
    splitter  = CardinalitySplitter(nmax=20)
    Xclt = Yclt = ClusterTree(X,splitter)
    adm       = StrongAdmissibilityStd(eta=3)
    rtol      = 1e-5
    comp      = PartialACA(rtol=rtol)
    # Laplace
    for threads in (true,false)
        K         = LaplaceMatrix(X,Y)
        H         = HMatrix(K,Xclt,Yclt,adm,comp;threads,distributed=false)
        @test norm(Matrix(K) - Matrix(H;global_index=true)) < rtol*norm(Matrix(K))
        adjH      = adjoint(H)
        H_full    = Matrix(H)
        adjH_full = adjoint(H_full)
        @testset "getindex" begin
            @test H_full[8,64] ≈ H[8,64]
            @test H_full[:,666] ≈ H[:,666]
            @test adjH_full[:,666] ≈ adjH[:,666]
        end
        # Elastostatic
        K         = ElastostaticMatrix(X,Y,1.1,1.2)
        H         = HMatrix(K,Xclt,Yclt,adm,comp)
        @test norm(Matrix(K) - Matrix(H;global_index=true)) < rtol*norm(Matrix(K))
    end
end
