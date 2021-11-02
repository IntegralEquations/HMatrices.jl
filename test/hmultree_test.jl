using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

dir = @__DIR__
include(joinpath(dir,"kernelmatrix.jl"))

using HMatrices: RkMatrix, plan_hmul, execute!

Random.seed!(1)

m = 5000
n = 5000
r = 4

X = Y = HMatrices.points_on_sphere(m)
splitter  = GeometricMinimalSplitter(nmax=50)
Xclt = Yclt = ClusterTree(X,splitter)
adm       = StrongAdmissibilityStd(eta=3)
atol      = 1e-5
comp      = PartialACA(;atol)
K         = LaplaceMatrix(X,Y)
H         = HMatrix(K,Xclt,Yclt,adm,comp;threads=false,distributed=false)

H_full = Matrix(H)

##
comp = PartialACA(;atol=1e-8)
exact = H_full + α*H_full*H_full
C  = deepcopy(H);
@test HMatrices.isclean(C)
hmultree = plan_hmul(C,H,H,α,1)
execute!(hmultree,comp)
@show norm(Matrix(C) - exact)
@test norm(Matrix(C) - exact) < 1e-4
@test HMatrices.isclean(C)
