using BEAST
using HMatrices
using CompScienceMeshes
using StaticArrays
using LinearAlgebra
using Test

@testset "BEAST KernelMatrix" begin
    ΓX = meshrectangle(1.0, 1.0, 0.2)
    ΓY = translate(ΓX, SVector(3.0, 0.0, 0.0))

    X = lagrangecxd0(ΓX)
    Y = lagrangecxd0(ΓY)
    op = Helmholtz3D.singlelayer()
    Kref = assemble(op, X, Y)

    #BEAST kernel matrix
    K = KernelMatrix(op, X, Y)

    @test Kref[1, 1] == K[1, 1]
    @test Kref[2, 1] == K[2, 1]
    @test Kref[1, 2] == K[1, 2]
    @test Kref[2:5, 5:7] == K[2:5, 5:7]
    @test size(Kref) == size(K)

    splitter = HMatrices.GeometricSplitter(4)
    rowtree = ClusterTree(X.pos, splitter)
    coltree = ClusterTree(Y.pos, splitter)

    Kperm = HMatrices.PermutedMatrix(
        K,
        HMatrices.loc2glob(rowtree),
        HMatrices.loc2glob(coltree),
    )

    out = zeros(eltype(Kref), 1:size(K, 1), 1:size(K, 2))
    HMatrices.getblock!(out, K, 1:size(K, 1), 1:size(K, 2))
    outperm = zeros(eltype(Kref), 1:size(K, 1), 1:size(K, 2))
    HMatrices.getblock!(outperm, Kperm, 1:size(K, 1), 1:size(K, 2))

    @test out != outperm
    @test out[Kperm.rowperm, Kperm.colperm] == outperm
end