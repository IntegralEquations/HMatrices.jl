using HMatrices
using StaticArrays
using Test
using HMatrices: GeometricMinimalSplitter, PrincipalComponentSplitter, GeometricSplitter, DyadicSplitter
using HMatrices: CardinalitySplitter, ClusterTree

function test_cluster_tree(clt)
    bbox = clt.bounding_box
    for iloc in clt.loc_idxs
        iglob = clt.loc2glob[iloc]
        x     = clt.points[iglob]
        x ∈ bbox || (return false)
    end
    for  child in clt.children
        test_cluster_tree(child) || (return false)
    end
    return true
end

@testset "ClusterTree" begin
    @testset "1d" begin
        points    = SVector.([4,3,1,2,5,-1.0])
        splitter  = GeometricSplitter(nmax=1)
        clt = ClusterTree(points,splitter)
        @test sortperm(points) == clt.loc2glob
        splitter  = GeometricMinimalSplitter(nmax=1)
        clt = ClusterTree(points,splitter)
        @test sortperm(points) == clt.loc2glob
        splitter  = CardinalitySplitter(nmax=1)
        clt = ClusterTree(points,splitter)
        @test sortperm(points) == clt.loc2glob
        splitter  = PrincipalComponentSplitter(nmax=1)
        clt = ClusterTree(points,splitter)
        @test sortperm(points) == clt.loc2glob
        splitter  = DyadicSplitter(nmax=1)
        clt = ClusterTree(points,splitter)
        @test sortperm(points) == clt.loc2glob
    end

    @testset "2d" begin
        points    = rand(SVector{2,Float64},1000)
        splitter  = GeometricSplitter(nmax=1)
        clt = ClusterTree(points,splitter)
        @test test_cluster_tree(clt)
        splitter  = GeometricMinimalSplitter(nmax=32)
        clt = ClusterTree(points,splitter)
        @test test_cluster_tree(clt)
        splitter   = CardinalitySplitter(nmax=32)
        clt = ClusterTree(points,splitter)
        @test test_cluster_tree(clt)
        splitter   = PrincipalComponentSplitter(nmax=32)
        clt = ClusterTree(points,splitter)
        @test test_cluster_tree(clt)
        splitter   = DyadicSplitter(nmax=32)
        clt = ClusterTree(points,splitter)
        @test test_cluster_tree(clt)
    end

    @testset "3d" begin
        points = rand(SVector{3,Float64},1000)
        splitter  = GeometricSplitter(nmax=32)
        clt = ClusterTree(points,splitter)
        @test test_cluster_tree(clt)
        splitter   = GeometricMinimalSplitter(nmax=32)
        clt = ClusterTree(points,splitter)
        @test test_cluster_tree(clt)
        splitter  = CardinalitySplitter(nmax=32)
        clt = ClusterTree(points,splitter)
        @test test_cluster_tree(clt)
        splitter   = PrincipalComponentSplitter(nmax=32)
        clt = ClusterTree(points,splitter)
        @test test_cluster_tree(clt)
        splitter   = DyadicSplitter(nmax=32)
        clt = ClusterTree(points,splitter)
        @test test_cluster_tree(clt)
    end

end
