using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays
using BenchmarkTools
using DataFlowTasks

using HMatrices

using GraphViz
using CairoMakie 

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

Random.seed!(1)

m = 5000
T = Float64
X = points_on_sphere(m)
Y = X

K = laplace_matrix(X, X)
K1 = laplace_matrix(X,X)

splitter = CardinalitySplitter(; nmax = 50)
Xclt = ClusterTree(X, splitter)
Yclt = ClusterTree(Y, splitter)
adm = StrongAdmissibilityStd(3)
comp = PartialACA(; atol = 1e-10)

function run(threads, dataflowtasks)
    println("run (threads=$threads, dataflowtasks=$dataflowtasks)")
    H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads, distributed = false)
    println(H)
    hlu = lu(H; atol = 1e-10, threads, dataflowtasks)
    y = rand(m)
    M = Matrix(K)
    exact = M \ y
    approx = hlu \ y
    @test norm(exact - approx, Inf) < 1e-10
    # test multiplication by checking if the solution is correct
    @test hlu.L * (hlu.U * approx) ≈ y
end

log_info = DataFlowTasks.@log run(true, true)
GraphViz.Graph(log_info)
plot(log_info; categories=["lu", "ldiv", "rdiv", "hmul"])
