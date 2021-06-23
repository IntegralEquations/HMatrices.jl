using AbstractTrees
using Test
using HMatrices
using StaticArrays
using HScheduler
using LinearAlgebra
using ComputationalResources

const HS = HScheduler

const HM = HMatrices
HM.debug()
using HMatrices: CardinalitySplitter, ClusterTree, RkMatrix

m,n  = 100_000,100_000
X    = rand(SVector{3,Float64},m)
Y    = [rand(SVector{3,Float64}) .+ 1e-5 for _  in 1:n]
splitter  = CardinalitySplitter(nmax=200)
Xclt      = ClusterTree(X,splitter)
Yclt      = ClusterTree(Y,splitter)
adm       = HMatrices.StrongAdmissibilityStd(eta=3)
rtol      = 1e-5
comp      = HMatrices.PartialACA(rtol=rtol)
sch       = HS.Scheduler(lmax=200)
HS.worker_init(sch)
# Laplace
K         = HMatrices.LaplaceMatrix(X,Y)
H = HMatrix(sch,K,Xclt,Yclt,adm,comp)
sch
# HM.@hprofile H = HMatrix(CPU1(),K,Xclt,Yclt,adm,comp)
# build edges
# @timev HS._build_edges_transitive!(tg);
# fill in data
# @elapsed HS._build_data_dependency!(tg)
# @timev HS._build_data_dependency!(tg);
