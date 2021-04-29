using StaticArrays
using HMatrices
using Test
using LinearAlgebra

using HMatrices: HyperRectangle, center, diameter, radius

@testset "HyperRectangle" begin
    low_corner  = SVector(0.0,0.0)
    high_corner = SVector(1.0,2.0)
    mid         = (low_corner + high_corner)/2
    rec = HyperRectangle(low_corner,high_corner)
    @test mid == center(rec)
    @test high_corner ∈ rec
    @test low_corner ∈ rec
    @test mid ∈ rec
    @test !in(high_corner + SVector(1,1),rec)
    rec1, rec2 = split(rec)
    @test low_corner ∈ rec1
    @test high_corner ∈ rec2
    @test !(low_corner ∈ rec2)
    @test !(high_corner ∈ rec1)
    @test diameter(rec) == sqrt(1^2 + 2^2)
    @test radius(rec) == sqrt(1^2 + 2^2)/2
    # bbox
    pts = SVector{2,Float64}[]
    for x=-1:0.1:1
        for y=-1:0.1:1
            push!(pts,SVector(x,y))
        end
    end
    @test HyperRectangle(pts) == HyperRectangle(SVector(-1.,-1),SVector(1,1.))
end
