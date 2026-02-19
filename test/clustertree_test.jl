using Test
using StaticArrays
using HMatrices

# recursively check that all point in a cluster tree are in the bounding box
function test_cluster_tree(clt)
    bbox = HMatrices.container(clt)
    for iloc in HMatrices.index_range(clt)
        x = HMatrices.root_elements(clt)[iloc]
        x ∈ bbox || (return false)
    end
    if !HMatrices.isroot(clt)
        clt ∈ clt.parentnode.children || (return false)
    end
    if !HMatrices.isleaf(clt)
        for child in clt.children
            test_cluster_tree(child) || (return false)
        end
    end
    return true
end

@testset "ClusterTree" begin
    @testset "1d" begin
        points = SVector.([4, 3, 1, 2, 5, -1.0])
        splitter = HMatrices.GeometricSplitter(; nmax = 1)
        clt = HMatrices.ClusterTree(points, splitter)
        @test sortperm(points) == clt.loc2glob
        splitter = HMatrices.CardinalitySplitter(; nmax = 1)
        clt = HMatrices.ClusterTree(points, splitter)
        @test sortperm(points) == clt.loc2glob
        splitter = HMatrices.PrincipalComponentSplitter(; nmax = 1)
        clt = HMatrices.ClusterTree(points, splitter)
        @test sortperm(points) == clt.loc2glob
    end

    @testset "2d" begin
        points = rand(SVector{2, Float64}, 1000)
        splitter = HMatrices.GeometricSplitter(; nmax = 1)
        clt = HMatrices.ClusterTree(points, splitter)
        @test test_cluster_tree(clt)
        splitter = HMatrices.CardinalitySplitter(; nmax = 32)
        clt = HMatrices.ClusterTree(points, splitter)
        @test test_cluster_tree(clt)
        splitter = HMatrices.PrincipalComponentSplitter(; nmax = 32)
        clt = HMatrices.ClusterTree(points, splitter)
        @test test_cluster_tree(clt)
    end

    @testset "3d" begin
        points = rand(SVector{3, Float64}, 1000)
        splitter = HMatrices.GeometricSplitter(; nmax = 32)
        clt = HMatrices.ClusterTree(points, splitter)
        @test test_cluster_tree(clt)
        splitter = HMatrices.CardinalitySplitter(; nmax = 32)
        clt = HMatrices.ClusterTree(points, splitter)
        @test test_cluster_tree(clt)
        splitter = HMatrices.PrincipalComponentSplitter(; nmax = 32)
        clt = HMatrices.ClusterTree(points, splitter)
        @test test_cluster_tree(clt)
    end

    @testset "3d + threads" begin
        threads = true
        points = rand(SVector{3, Float64}, 1000)
        splitter = HMatrices.GeometricSplitter(; nmax = 32)
        clt = HMatrices.ClusterTree(points, splitter; threads)
        @test test_cluster_tree(clt)
        splitter = HMatrices.CardinalitySplitter(; nmax = 32)
        clt = HMatrices.ClusterTree(points, splitter; threads)
        @test test_cluster_tree(clt)
        splitter = HMatrices.PrincipalComponentSplitter(; nmax = 32)
        clt = HMatrices.ClusterTree(points, splitter; threads)
        @test test_cluster_tree(clt)
    end
end
