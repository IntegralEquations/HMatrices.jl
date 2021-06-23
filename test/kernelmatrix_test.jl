using Test
using HMatrices
using StaticArrays
using ComputationalResources
using LinearAlgebra
using BenchmarkTools

using HMatrices: LaplaceMatrix, LaplaceMatrixVec, HelmholtzMatrix, HelmholtzMatrixVec

@testset "Laplace" begin
    m,n = 200,200
    X = rand(SVector{3,Float64},m)
    Y = rand(SVector{3,Float64},n)
    K = LaplaceMatrix(X,Y)
    K_vec = LaplaceMatrixVec(X,Y)

    # test whole matrix
    @test norm(K[1:m,1:n]-K_vec[1:m,1:n],Inf) < 1e-10
    # and a sub block
    @test norm(K[5:m,6:n]-K_vec[5:m,6:n],Inf) < 1e-10
    # and a row range
    @test norm(K[4,5:n-5]-K_vec[4,5:n-5],Inf) < 1e-10
    # and a col
    @test norm(K[5:m-5,4]-K_vec[5:m-5,4],Inf) < 1e-10

    @btime ($K)[1:m,1:n];
    @btime ($K_vec)[1:m,1:n];
    # @btime ($K)[1,1:n];
    # @btime ($K_vec)[1,1:n];
    # @btime ($K)[1:m,1];
    # @btime ($K_vec)[1:m,1];

end

@testset "Helmholtz" begin
    m,n = 200,200
    k = 3.0
    X = rand(SVector{3,Float64},m)
    Y = rand(SVector{3,Float64},n)
    K     = HelmholtzMatrix(X,Y,k)
    K_vec = HelmholtzMatrixVec(X,Y,k)

    # test whole matrix
    @test norm(K[1:m,1:n]-K_vec[1:m,1:n],Inf) < 1e-10
    # and a sub block
    @test norm(K[5:m,6:n]-K_vec[5:m,6:n],Inf) < 1e-10
    # and a row range
    @test norm(K[4,5:n-5]-K_vec[4,5:n-5],Inf) < 1e-10
    # and a col
    @test norm(K[5:m-5,4]-K_vec[5:m-5,4],Inf) < 1e-10

    I,J = 1:m, 1:n
    @btime ($K)[$I,$J];
    @btime ($K_vec)[$I,$J];
    @btime ($K)[1,$J];
    @btime ($K_vec)[1,$J];
    @btime ($K)[$I,1];
    @btime ($K_vec)[$I,1];

end
