using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

using HMatrices: RkMatrix, points_on_sphere

Random.seed!(1)

m = 1000
T = Float64
X    = points_on_sphere(m)
# X    = rand(SVector{3,Float64},m)
Y    = X
struct ExponentialKernel <: AbstractMatrix{Float64}
    X::Vector{SVector{3,Float64}}
    Y::Vector{SVector{3,Float64}}
end
function Base.getindex(K::ExponentialKernel,i::Int,j::Int)
    x,y = K.X[i], K.Y[j]
    exp(-norm(x-y))
end
Base.size(K::ExponentialKernel) = length(K.X), length(K.Y)
K = ExponentialKernel(X,X)

splitter  = CardinalitySplitter(nmax=50)
Xclt      = ClusterTree(X,splitter)
Yclt      = ClusterTree(Y,splitter)
H = HMatrix(K,Xclt,Yclt;threads=false,distributed=false)
H_full      = Matrix(H)

@testset "ldiv!" begin
    B = rand(m,2)
    R = RkMatrix(rand(m,3),rand(m,3))
    R_full = Matrix(R)

    ## 3.1
    exact  = ldiv!(UnitLowerTriangular(H_full),copy(B))
    approx = ldiv!(UnitLowerTriangular(H),copy(B))
    @test exact ≈ approx

    exact  = ldiv!(UpperTriangular(H_full),copy(B))
    approx = ldiv!(UpperTriangular(H),copy(B))
    @test exact ≈ approx

    ## 3.2
    exact  = ldiv!(UnitLowerTriangular(H_full),copy(R_full))
    approx = ldiv!(UnitLowerTriangular(H),copy(R))
    @test exact ≈ Matrix(approx)

    ## 3.3
    compressor = PartialACA(;atol=1e-8)
    exact      = ldiv!(UnitLowerTriangular(H_full),copy(H_full))
    approx     = ldiv!(UnitLowerTriangular(H),deepcopy(H),compressor)
    @test exact ≈ Matrix(approx)
end

@testset "rdiv!" begin
    B = rand(2,m)
    R = RkMatrix(rand(m,3),rand(m,3))
    R_full = Matrix(R)
    # 3.1
    exact  = rdiv!(copy(B),UpperTriangular(H_full))
    approx = rdiv!(copy(B),UpperTriangular(H))
    @test exact ≈ approx

    ## 3.2
    exact  = rdiv!(copy(R_full),UpperTriangular(H_full))
    approx = rdiv!(copy(R),UpperTriangular(H))
    @test exact ≈ approx

    ## 3.3
    compressor = PartialACA(;atol=1e-10)
    exact  = rdiv!(deepcopy(H_full),UpperTriangular(H_full))
    approx = rdiv!(deepcopy(H),UpperTriangular(H),compressor)
    @test exact ≈ Matrix(approx)
end
