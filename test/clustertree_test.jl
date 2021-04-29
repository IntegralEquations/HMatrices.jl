using HMatrices
using StaticArrays
using Test
using HMatrices: GeometricSplitter, GeometricMinimalSplitter
using HMatrices: CardinalitySplitter, ClusterTree, QuadOctSplitter

@testset "ClusterTree" begin

    @testset "2d test" begin
        data = rand(SVector{2,Float64},1000)
        splitter   = GeometricSplitter(nmax=32)
        clt = ClusterTree(;data,splitter,reorder=false)
        @test clt.data == data[clt.perm]
        splitter   = GeometricMinimalSplitter(nmax=32)
        clt = ClusterTree(;data,splitter,reorder=false)
        @test clt.data == data[clt.perm]
        splitter   = CardinalitySplitter(nmax=32)
        clt = ClusterTree(;data,splitter,reorder=false)
        @test clt.data == data[clt.perm]
        splitter   = QuadOctSplitter(nmax=32)
        clt = ClusterTree(;data,splitter,reorder=false)
        @test clt.data == data[clt.perm]
    end

    @testset "3d tests" begin
        data = rand(SVector{3,Float64},1000)
        splitter   = GeometricSplitter(nmax=32)
        clt = ClusterTree(;data,splitter,reorder=false)
        @test clt.data == data[clt.perm]
        splitter   = GeometricMinimalSplitter(nmax=32)
        clt = ClusterTree(;data,splitter,reorder=false)
        @test clt.data == data[clt.perm]
        splitter   = CardinalitySplitter(nmax=32)
        clt = ClusterTree(;data,splitter,reorder=false)
        @test clt.data == data[clt.perm]
        splitter   = QuadOctSplitter(nmax=32)
        clt = ClusterTree(;data,splitter,reorder=false)
        @test clt.data == data[clt.perm]
    end

end
