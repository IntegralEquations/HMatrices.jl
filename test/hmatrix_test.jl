using Test
using HMatrices
using StaticArrays
using ComputationalResources
using LinearAlgebra

const HM = HMatrices
# HM.debug()
using HMatrices: CardinalitySplitter, ClusterTree, RkMatrix

@testset "CPU1" begin
    m,n  = 2000,2000
    X    = rand(SVector{3,Float64},m)
    Y    = [rand(SVector{3,Float64}) .+ 1e-5 for _  in 1:n]
    splitter  = CardinalitySplitter(nmax=200)
    Xclt      = ClusterTree(X,splitter)
    Yclt      = ClusterTree(Y,splitter)
    adm       = HMatrices.StrongAdmissibilityStd(eta=3)
    rtol      = 1e-5
    comp      = HMatrices.PartialACA(rtol=rtol)
    resource  = CPU1()
    # Laplace
    K         = HMatrices.LaplaceMatrix(X,Y)
    H         = HMatrix(resource,K,Xclt,Yclt,adm,comp)
    # test below by computing action on vector
    @testset "gemv" begin
        T    = eltype(H)
        a,b  = 2,2
        x    = rand(T,n)
        y    = rand(T,m)
        ye   = b*y+a*K*x
        mul!(y,H,x,a,b)
        @test norm(y-ye) < rtol*norm(ye)
    end
    # Helmholtz
    K         = HMatrices.HelmholtzMatrix(X,Y,1.2)
    H         = HMatrix(resource,K,Xclt,Yclt,adm,comp)
    # test below by computing action on vector
    @testset "gemv" begin
        T    = eltype(H)
        a,b  = 2,2
        x    = rand(T,n)
        y    = rand(T,m)
        ye   = b*y+a*K*x
        mul!(y,H,x,a,b)
        @test norm(y-ye) < rtol*norm(ye)
    end
end
