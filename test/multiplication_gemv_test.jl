using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

dir = @__DIR__
include(joinpath(dir,"kernelmatrix.jl"))

Random.seed!(1)

m = 5000
n = 5000
r = 4

X = Y = rand(SVector{3,Float64},m)
splitter  = CardinalitySplitter(nmax=100)
Xclt  = Yclt     = ClusterTree(X,splitter)
adm       = StrongAdmissibilityStd(eta=3)
rtol      = 1e-10
comp      = PartialACA(rtol=rtol)
K         = LaplaceMatrix(X,Y)
H         = HMatrix(K,Xclt,Yclt,adm,comp;threads=false,distributed=false)
H_full = Matrix(H)
T = eltype(H)
x = rand(T,size(H,2))
exact = Matrix(K)*x
@test norm(H*x - exact) < rtol*norm(exact)
