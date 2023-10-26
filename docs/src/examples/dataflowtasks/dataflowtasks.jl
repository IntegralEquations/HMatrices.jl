# cd(@__DIR__)             #src
# import Pkg               #src
# Pkg.activate("../../..") #src

#=

# Threaded Parallelism with [`DataFlowTasks`](@ref)

In this example we explore how `HMatrices` makes use of `DataFlowTasks` to
exploit shared memory parallelism. In particular, we will go into some detail as
to how to pipeline code so that...

Let us begin by creating a (sufficiently large) matrix for representing
=#

using HMatrices, LinearAlgebra, StaticArrays, IterativeSolvers
using DataFlowTasks

capacity = 500
sch = DataFlowTasks.JuliaScheduler(capacity)
DataFlowTasks.setscheduler!(sch)

DataFlowTasks.force_sequential(false)
DataFlowTasks.force_linear_dag() = false

BLAS.set_num_threads(1)

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

N = 50_000
nmax = 200
eta = 3
rtol = 1e-5
radius = 1

X = points_on_cylinder(N, radius)
splitter = HMatrices.CardinalitySplitter(nmax)
Xclt = HMatrices.ClusterTree(X, splitter)
Xp = HMatrices.root_elements(Xclt)
adm = HMatrices.StrongAdmissibilityStd(eta)
comp = HMatrices.PartialACA(; rtol)

step = 1.75 * π * radius / sqrt(N)
k = 2 * π / (10 * step) # 10 pts per wavelength

K = HelmholtzMatrixVec(X, X, k)

threads = false

GC.gc()
@time H = assemble_hmatrix(K, Xclt, Xclt; adm, comp, threads = true, distributed = false)
Hc = deepcopy(H)
GC.gc()
@time F = lu!(Hc; atol = 1e-4, threads)
Hc = deepcopy(H)
GC.gc()
loginfo = DataFlowTasks.@log lu!(Hc; atol = 1e-4, threads)
