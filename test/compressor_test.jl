using HMatrices
using Test
using StaticArrays
using LinearAlgebra
using HMatrices: ACA, PartialACA, TSVD, RkMatrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

using Random
Random.seed!(1)

@testset "Scalar" begin
    T = ComplexF64
    m, n = 100, 100
    X = rand(SVector{3,Float64}, m)
    Y = map(i -> SVector(10, 0, 0) + rand(SVector{3,Float64}), 1:n)
    K = helmholtz_matrix(X, Y, 1.0)
    M = Matrix(K)
    irange, jrange = 1:m, 1:n
    @testset "aca_full" begin
        atol = 1e-5
        aca = ACA(; atol = atol)
        R = aca(K, irange, jrange)
        @test norm(Matrix(R) - M) < atol
        rtol = 1e-5
        aca = ACA(; rtol = rtol)
        R = aca(M, irange, jrange)
        norm(Matrix(R) - M)
        @test norm(Matrix(R) - M) < rtol * norm(M)
        r = 10
        aca = ACA(; rank = r)
        R = aca(M, irange, jrange)
        @test rank(R) == r
    end
    @testset "aca_partial" begin
        atol = 1e-5
        aca = PartialACA(; atol = atol)
        R = aca(M, irange, jrange)
        @test norm(Matrix(R) - M) < atol
        rtol = 1e-5
        aca = PartialACA(; rtol = rtol)
        R = aca(M, irange, jrange)
        @test norm(Matrix(R) - M) < rtol * norm(M)
        r = 10
        aca = PartialACA(; rank = r)
        R = aca(M, irange, jrange)
        @test rank(R) == r

        # test fast update of frobenius norm
        m, n = 10000, 1000
        r = 10
        T = ComplexF64
        A = [rand(T, m) for _ in 1:r]
        B = [rand(T, n) for _ in 1:r]
        R = RkMatrix(A, B)
        old_norm = norm(Matrix(R), 2)
        push!(A, rand(T, m))
        push!(B, rand(T, n))
        Rnew = RkMatrix(A, B)
        new_norm = norm(Matrix(Rnew), 2)
        @test new_norm ≈ HMatrices._update_frob_norm(old_norm, A, B)

        # test simple case where things are not compressible
        A = rand(2, 2)
        comp = PartialACA(; rtol = 1e-5)
        @test comp(A, 1:2, 1:2) ≈ A
    end
    @testset "truncated svd" begin
        atol = 1e-5
        tsvd = TSVD(; atol = atol)
        R = tsvd(M, irange, jrange)
        # the inequality below is guaranteed to be true  for  the spectral norm
        # i.e. (the `opnorm` with `p=2`).
        @test opnorm(Matrix(R) - M) < atol
        rtol = 1e-5
        tsvd = TSVD(; rtol = rtol)
        R = tsvd(M, irange, jrange)
        @test opnorm(Matrix(R) - M) < rtol * opnorm(M)
        r = 10
        tsvd = TSVD(; rank = r)
        R = tsvd(M, irange, jrange)
        @test rank(R) == r
    end
end

@testset "Tensorial" begin
    T = SMatrix{3,3,ComplexF64,9}
    # T = SMatrix{3,3,Float64,9}
    m, n = 100, 100
    X = rand(SVector{3,Float64}, m)
    Y = map(i -> SVector(10, 0, 0) + rand(SVector{3,Float64}), 1:n)
    K = elastosdynamic_matrix(X, Y, 1.0, 2.0, 1.0, 1.0)
    # K = ElastostaticMatrix(X,Y,1.0,2.0)
    M = Matrix(K)
    irange, jrange = 1:m, 1:n
    @testset "aca_full" begin
        atol = 1e-5
        aca = ACA(; atol = atol)
        R = aca(K, irange, jrange)
        @test norm(Matrix(R) - M) < atol
        rtol = 1e-5
        aca = ACA(; rtol = rtol)
        R = aca(M, irange, jrange)
        norm(Matrix(R) - M)
        @test norm(Matrix(R) - M) < rtol * norm(M)
        r = 10
        aca = ACA(; rank = r)
        R = aca(M, irange, jrange)
        @test rank(R) == r
    end
    @testset "aca_partial" begin
        atol = 1e-5
        aca = PartialACA(; atol = atol)
        R = aca(M, irange, jrange)
        @test norm(Matrix(R) - M) < atol
        rtol = 1e-5
        aca = PartialACA(; rtol = rtol)
        R = aca(M, irange, jrange)
        @test norm(Matrix(R) - M) < rtol * norm(M)
        r = 10
        aca = PartialACA(; rank = r)
        R = aca(M, irange, jrange)
        @test rank(R) == r

        # test fast update of frobenius norm
        m, n = 10000, 1000
        r = 10
        T = ComplexF64
        A = [rand(T, m) for _ in 1:r]
        B = [rand(T, n) for _ in 1:r]
        R = RkMatrix(A, B)
        old_norm = norm(Matrix(R), 2)
        push!(A, rand(T, m))
        push!(B, rand(T, n))
        Rnew = RkMatrix(A, B)
        new_norm = norm(Matrix(Rnew), 2)
        @test new_norm ≈ HMatrices._update_frob_norm(old_norm, A, B)

        # test simple case where things are not compressible
        A = rand(2, 2)
        comp = PartialACA(; rtol = 1e-5)
        @test comp(A, 1:2, 1:2) ≈ A
    end
end
