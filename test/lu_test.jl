using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

using HMatrices: RkMatrix, points_on_sphere

Random.seed!(1)

m = 2000
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

##
exact = lu(H_full)
comp = PartialACA(;atol=1e-8)
approx   = lu(H,comp)
@test norm(exact.L - Matrix(approx.L),Inf) < 1e-6
@test norm(exact.U - Matrix(approx.U),Inf) < 1e-6
