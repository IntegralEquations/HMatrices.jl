using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

HMatrices.debug(false)
using HMatrices: RkMatrix

Random.seed!(1)

m = 5000
n = 5000
r = 4

T = Float64

R = RkMatrix(rand(m,r),rand(n,r))
F = rand(T,m,n)

X    = rand(SVector{3,Float64},m)
Y    = [rand(SVector{3,Float64}) .+ (1,0,0) for _  in 1:n]
splitter  = CardinalitySplitter(nmax=100)
Xclt      = ClusterTree(X,splitter)
Yclt      = ClusterTree(Y,splitter)
adm       = StrongAdmissibilityStd(eta=3)
rtol      = 1e-5
comp      = PartialACA(rtol=rtol)
K         = HMatrices.LaplaceMatrix(X,Y)
H         = HMatrix(K,Xclt,Yclt,adm,comp;threads=false,distributed=false)

H_full = Matrix(H)
R_full = Matrix(R)

## Test the various posthooks that one can pass to the hierarchical mul! method
α,β = rand(),rand()

## identity
compressor = identity
C  = deepcopy(H);
tmp = β*H_full + α*H_full*H_full
mul!(C,H,H,α,β,compressor)
@test Matrix(C) ≈ tmp

## compress blocks after creation and flush to leaves
comp = TSVD(;atol=1e-7)
C  = deepcopy(H);
tmp = β*H_full + α*H_full*H_full
mul!(C,H,H,α,β,comp)
@test norm(Matrix(C) - tmp) < 1e-5
