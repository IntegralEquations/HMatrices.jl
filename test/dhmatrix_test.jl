using Distributed
addprocs(2; exeflags = `--project=$(Base.active_project())`)

using Test
using StaticArrays
using LinearAlgebra
@everywhere using HMatrices
@everywhere include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))
# the kernel must be defined in `Main`, where the workers can deserialize it
const points_on_cylinder = Main.points_on_cylinder
const laplace_matrix = Main.laplace_matrix

@testset "Assemble" begin
    m = 2_000
    X = Y = points_on_cylinder(m, 1)
    splitter = CardinalitySplitter(; nmax = 100)
    Xclt = Yclt = ClusterTree(X, splitter)
    adm = StrongAdmissibilityStd(; eta = 3)
    comp = PartialACA(; atol = 1.0e-6)
    K = laplace_matrix(X, Y)
    H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, distributed = false, threads = false)
    Hd = assemble_hmatrix(K, Xclt, Yclt; adm, comp, distributed = true, threads = false)
    @test Hd isa HMatrices.DHMatrix
    @test HMatrices.isclean(Hd)
    @test length(HMatrices.leaves(Hd)) == 2
    x = rand(m)
    @test H * x ≈ Hd * x
    y = rand(m)
    @test mul!(copy(y), H, x, 2, 3; threads = false) ≈ mul!(copy(y), Hd, x, 2, 3; threads = false)
    @test contains(sprint(show, MIME("text/plain"), Hd), "leaf 2 on process")
end

rmprocs(workers())
