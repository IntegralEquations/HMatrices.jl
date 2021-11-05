using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays
using HMatrices: RkMatrix

dir = @__DIR__
include(joinpath(dir,"kernelmatrix.jl"))

N,r     = 500,3
atol    = 1e-6
X  = Y     = rand(SVector{3,Float64},N)
splitter = CardinalitySplitter(nmax=50)
Xclt = Yclt  = ClusterTree(X,splitter)
adm       = StrongAdmissibilityStd(eta=3)
rtol      = 1e-5
comp      = PartialACA(rtol=rtol)
K         = HelmholtzMatrix(X,Y,1.3)
H         = HMatrix(K,Xclt,Yclt,adm,comp;threads=false,distributed=false)
T         = eltype(H)
_H   = Matrix(H)
R    = RkMatrix(rand(T,N,r),rand(T,N,r))
_R   = Matrix(R)
M    = rand(T,N,N)
_M   = Matrix(M)
a    = rand()
@testset "+" begin
    @test Matrix(M+R) ≈ _M + _R
    @test Matrix(M+H) ≈ _M + _H
    @test Matrix(R+M) ≈ _R + _M
    @test Matrix(R+R) ≈ _R + _R
    @test Matrix(R+H) ≈ _R + _H
    @test Matrix(H+M) ≈ _H + _M
    @test Matrix(H+R) ≈ _H + _R
    @test Matrix(H+H) ≈ _H + _H
end
@testset "axpy!" begin
    @test axpy!(a,M,deepcopy(R)) ≈ a*_M + _R
    @test axpy!(a,M,deepcopy(H)) ≈ a*_M + _H
    @test axpy!(a,R,deepcopy(M)) ≈ a*_R + _M
    @test axpy!(a,R,deepcopy(R)) ≈ a*_R + _R
    @test axpy!(a,R,deepcopy(H)) ≈ a*_R + _H
    @test axpy!(a,H,deepcopy(M)) ≈ a*_H + _M
    @test axpy!(a,H,deepcopy(R)) ≈ a*_H + _R
    @test axpy!(a,H,deepcopy(H)) ≈ a*_H + _H
end
