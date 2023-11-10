using Test
using HMatrices
using StaticArrays

@testset "HyperRectangle tests" begin
    low_corner = SVector(0.0, 0.0)
    high_corner = SVector(1.0, 2.0)
    mid = (low_corner + high_corner) / 2
    rec = HMatrices.HyperRectangle(low_corner, high_corner)
    @test mid == HMatrices.center(rec)
    @test high_corner ∈ rec
    @test low_corner ∈ rec
    @test mid ∈ rec
    @test !in(high_corner + SVector(1, 1), rec)
    rec1, rec2 = split(rec)
    @test low_corner ∈ rec1
    @test high_corner ∈ rec2
    @test !(low_corner ∈ rec2)
    @test !(high_corner ∈ rec1)
    @test HMatrices.diameter(rec) == sqrt(1^2 + 2^2)
    @test HMatrices.radius(rec) == sqrt(1^2 + 2^2) / 2
    # bbox
    pts = SVector{2,Float64}[]
    for x in -1:0.1:1
        for y in -1:0.1:1
            push!(pts, SVector(x, y))
        end
    end
    @test HMatrices.bounding_box(pts) ==
          HMatrices.HyperRectangle(SVector(-1.0, -1), SVector(1, 1.0))
    @test HMatrices.bounding_box(pts, true) ==
          HMatrices.HyperRectangle(SVector(-1.0, -1), SVector(1, 1.0))
    pts = SVector{2,Float64}[]
    for x in -1:0.1:1
        for y in -1:0.1:2
            push!(pts, SVector(x, y))
        end
    end
    @test HMatrices.bounding_box(pts) ==
          HMatrices.HyperRectangle(SVector(-1.0, -1), SVector(1, 2.0))
    @test HMatrices.bounding_box(pts, true) ==
          HMatrices.HyperRectangle(SVector(-1.5, -1), SVector(3 / 2, 2.0))
    rec1 = HMatrices.HyperRectangle(SVector(0, 0), SVector(1, 1))
    rec2 = HMatrices.HyperRectangle(SVector(2, 0), SVector(3, 1))
    @test HMatrices.distance(rec1, rec2) ≈ 1
    x = SVector(0.5, 0.5)
    rec2 = HMatrices.HyperRectangle(SVector(2, 2), SVector(3, 3))
    HMatrices.distance(rec1, rec2) ≈ sqrt(2)
end
