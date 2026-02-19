using HMatrices
using Test
using StaticArrays
using LinearAlgebra
using HMatrices:
    PartialACA,
    TSVD,
    ACAWithRecompression,
    ACAWithRecompressionBuffer,
    RkMatrix,
    VectorOfVectors,
    allocate_buffer

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

using Random
Random.seed!(1)

@testset "Scalar" begin
    T = ComplexF64
    m, n = 100, 100
    X = rand(SVector{3, Float64}, m)
    Y = map(i -> SVector(10, 0, 0) + rand(SVector{3, Float64}), 1:n)
    K = helmholtz_matrix(X, Y, 1.0)
    M = Matrix(K)
    irange, jrange = 1:m, 1:n
    @testset "aca_partial" begin
        atol = 1.0e-5
        aca = PartialACA(; atol = atol)
        R = aca(M, irange, jrange)
        @test norm(Matrix(R) - M) < atol
        rtol = 1.0e-5
        aca = PartialACA(; rtol = rtol)
        R = aca(M, irange, jrange)
        @test norm(Matrix(R) - M) < rtol * norm(M)
        r = 10
        aca = PartialACA(; rank = r)
        R = aca(M, irange, jrange)
        @test rank(R) == r

        # test zero matrix
        R = aca(zero(M), irange, jrange)
        @test norm(Matrix(R)) ≈ 0.0

        # test fast update of frobenius norm
        m, n = 10000, 1000
        r = 10
        A = VectorOfVectors(T, m, r)
        B = VectorOfVectors(T, n, r)
        A.data .= rand(T, m * r)
        B.data .= rand(T, n * r)
        old_norm = norm(Matrix(A) * adjoint(Matrix(B)), 2)
        a = HMatrices.newcol!(A)
        a .= rand(T, m)
        b = HMatrices.newcol!(B)
        b .= rand(T, n)
        new_norm = norm(Matrix(A) * adjoint(Matrix(B)), 2)
        @test new_norm ≈ HMatrices._update_frob_norm(old_norm, A, B)

        # test simple case where things are not compressible
        A = rand(2, 2)
        comp = PartialACA(; rtol = 1.0e-5)
        @test comp(A, 1:2, 1:2) ≈ A
    end
    @testset "aca_with_recompression" begin
        # accuracy: atol
        atol = 1.0e-5
        comp = ACAWithRecompression(PartialACA(; atol))
        R = comp(M, irange, jrange)
        @test norm(Matrix(R) - M) < 5 * atol
        # accuracy: rtol
        rtol = 1.0e-5
        comp = ACAWithRecompression(PartialACA(; rtol))
        R = comp(M, irange, jrange)
        @test norm(Matrix(R) - M) < rtol * norm(M)
        # rank cap is respected
        r = 10
        comp = ACAWithRecompression(PartialACA(; rank = r))
        R = comp(M, irange, jrange)
        @test rank(R) <= r
        # recompression should not increase rank vs PartialACA alone
        aca = PartialACA(; rtol = 1.0e-5)
        R_aca = aca(M, irange, jrange)
        R_recomp = ACAWithRecompression(aca)(M, irange, jrange)
        @test rank(R_recomp) <= rank(R_aca)
        # with explicit buffer: results should match no-buffer path
        comp_buf = ACAWithRecompression(PartialACA(; rtol = 1.0e-5))
        buf = ACAWithRecompressionBuffer(T)
        R_with_buf = comp_buf(M, irange, jrange, buf)
        R_no_buf = comp_buf(M, irange, jrange)
        @test rank(R_with_buf) == rank(R_no_buf)
        @test norm(Matrix(R_with_buf) - M) < 1.0e-5 * norm(M)
        # allocate_buffer dispatch
        @test allocate_buffer(PartialACA(), Float64) isa HMatrices.ACABuffer{Float64}
        @test allocate_buffer(ACAWithRecompression(PartialACA()), Float64) isa ACAWithRecompressionBuffer{Float64}
        @test allocate_buffer(TSVD(), Float64) === nothing
    end
    @testset "truncated svd" begin
        atol = 1.0e-5
        tsvd = TSVD(; atol = atol)
        R = tsvd(M, irange, jrange)
        # the inequality below is guaranteed to be true  for  the spectral norm
        # i.e. (the `opnorm` with `p=2`).
        @test opnorm(Matrix(R) - M) < atol
        rtol = 1.0e-5
        tsvd = TSVD(; rtol = rtol)
        R = tsvd(M, irange, jrange)
        @test opnorm(Matrix(R) - M) < rtol * opnorm(M)
        r = 10
        tsvd = TSVD(; rank = r)
        R = tsvd(M, irange, jrange)
        @test rank(R) == r
        # test recompression using QR-SVD
        R2 = RkMatrix(hcat(R.A, R.A), hcat(R.B, R.B))
        @test rank(R2) == 2r
        HMatrices.compress!(R2, tsvd)
        @test rank(R2) == r
    end
end

@testset "Tensorial" begin
    T = SMatrix{3, 3, ComplexF64, 9}
    # T = SMatrix{3,3,Float64,9}
    m, n = 100, 100
    X = rand(SVector{3, Float64}, m)
    Y = map(i -> SVector(10, 0, 0) + rand(SVector{3, Float64}), 1:n)
    K = elastosdynamic_matrix(X, Y, 1.0, 2.0, 1.0, 1.0)
    # K = ElastostaticMatrix(X,Y,1.0,2.0)
    M = Matrix(K)
    irange, jrange = 1:m, 1:n

    @testset "aca_partial" begin
        atol = 1.0e-5
        aca = PartialACA(; atol = atol)
        R = aca(M, irange, jrange)
        @test norm(Matrix(R) - M) < atol
        rtol = 1.0e-5
        aca = PartialACA(; rtol = rtol)
        R = aca(M, irange, jrange)
        @test norm(Matrix(R) - M) < rtol * norm(M)
        r = 10
        aca = PartialACA(; rank = r)
        R = aca(M, irange, jrange)
        @test rank(R) == r

        # test fast update of frobenius norm
        T = SMatrix{3, 3, Float64, 9}
        A = VectorOfVectors(T, m, r)
        B = VectorOfVectors(T, n, r)
        A.data .= rand(T, m * r)
        B.data .= rand(T, n * r)
        R = Matrix(A) * collect(adjoint(Matrix(B)))
        old_norm = norm(R, 2)
        a = HMatrices.newcol!(A)
        a .= rand(T, m)
        b = HMatrices.newcol!(B)
        b .= rand(T, n)
        R = Matrix(A) * collect(adjoint(Matrix(B)))
        new_norm = norm(R, 2)
        HMatrices._update_frob_norm(old_norm, A, B)
        @test_broken new_norm ≈ HMatrices._update_frob_norm(old_norm, A, B)
    end
end
