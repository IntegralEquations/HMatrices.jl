using HMatrices
using LinearAlgebra
using Test
using StaticArrays
using HMatrices: RkMatrix

@testset "RkMatrix" begin
    @testset "Scalar entries" begin
        m = 20
        n = 30
        r = 5
        A = rand(ComplexF64, m, r)
        a = rand(ComplexF64, m)
        B = rand(ComplexF64, n, r)
        b = rand(ComplexF64, n)
        R = RkMatrix(A, B)
        Ra = adjoint(R)
        M = A * adjoint(B)
        Ma = adjoint(M)

        ## basic tests
        @test size(R) == (m, n)
        @test rank(R) == r
        @test Matrix(R) ≈ M
        @test HMatrices.getcol!(a, R, 5) ≈ M[:, 5]
        @test HMatrices.getcol!(b, Ra, 5) ≈ Ma[:, 5]
    end

    @testset "Matrix entries" begin
        m = 20
        n = 30
        r = 10
        T = SMatrix{3, 3, ComplexF64, 9}
        A = rand(T, m, r)
        B = rand(T, n, r)
        R = RkMatrix(A, B)
        ## basic tests
        @test size(R) == (m, n)
        @test rank(R) == r
    end
end
