using HMatrices
using Test
using StaticArrays
using LinearAlgebra
using HMatrices:
    PartialACA,
    TSVD,
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

        # test zero matrix: no warning, since every row is exactly zero
        R = @test_logs aca(zero(M), irange, jrange)
        @test norm(Matrix(R)) ≈ 0.0
        # nor on a block approximated exactly, whose remaining rows are zero
        Z = zero(M)
        Z[:, 1] .= 1
        R = @test_logs aca(Z, irange, jrange)
        @test Matrix(R) == Z

        # rows much smaller than the others: the next row is taken from the residual
        # column, not by scanning the rows
        evals = Ref(0)
        Zs = zeros(200, 200)
        Zs[1:100, :] .= 1.0e-12 .* rand(100) * rand(200)'
        Zs[101:200, :] .= rand(100) * rand(200)'
        Kc = KernelMatrix((i, j) -> (evals[] += 1; Zs[i, j]), 1:200, 1:200)
        R = PartialACA(; rtol = 1.0e-14)(Kc)
        @test norm(Matrix(R) - Zs) < 1.0e-12 * norm(Zs)
        @test evals[] < 10_000

        # invalid tolerances
        for kw in ((; atol = -1.0), (; rank = 0), (; rtol = -1.0), (; rtol = NaN), (; atol = NaN))
            @test_throws ArgumentError PartialACA(; kw...)
            @test_throws ArgumentError TSVD(; kw...)
        end
        @test_throws ArgumentError PartialACA(; samples = -1)

        # a buffer left in use by an exception can be reused
        buf = allocate_buffer(aca, T)
        bad = KernelMatrix((i, j) -> i == 7 ? error("bad entry") : M[i, j], 1:m, 1:n)
        @test_throws ErrorException aca(bad, irange, jrange, buf)
        @test norm(Matrix(aca(M, irange, jrange, buf)) - M) < atol

        # zero rows use up the candidate rows without increasing the rank
        Z = zeros(T, 4, 5)
        Z[4, :] = rand(T, 5)
        R = PartialACA(; atol = 1.0e-10)(Z)
        @test Matrix(R) ≈ Z

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
    @testset "aca with recompression" begin
        # accuracy: atol
        atol = 1.0e-5
        comp = PartialACA(; atol, recompress = true)
        R = comp(M, irange, jrange)
        @test norm(Matrix(R) - M) < 5 * atol
        # accuracy: rtol
        rtol = 1.0e-5
        comp = PartialACA(; rtol, recompress = true)
        R = comp(M, irange, jrange)
        @test norm(Matrix(R) - M) < rtol * norm(M)
        # a rank larger than the block is recompressed through the dense matrix
        A, B = rand(T, 10, 15), rand(T, 30, 15)
        R = HMatrices.compress!(RkMatrix(copy(A), copy(B)), TSVD(; rtol = 1.0e-10))
        @test rank(R) == 10 && Matrix(R) ≈ A * B'
        # rank cap is respected
        r = 10
        comp = PartialACA(; rank = r, recompress = true)
        R = comp(M, irange, jrange)
        @test rank(R) <= r
        # recompression should not increase rank vs PartialACA alone
        aca = PartialACA(; rtol = 1.0e-5)
        R_aca = aca(M, irange, jrange)
        R_recomp = PartialACA(; rtol = 1.0e-5, recompress = true)(M, irange, jrange)
        @test rank(R_recomp) <= rank(R_aca)
        # with explicit buffer: results should match no-buffer path
        comp_buf = PartialACA(; rtol = 1.0e-5, recompress = true)
        buf = allocate_buffer(comp_buf, T)
        R_with_buf = comp_buf(M, irange, jrange, buf)
        R_no_buf = comp_buf(M, irange, jrange)
        @test rank(R_with_buf) == rank(R_no_buf)
        @test norm(Matrix(R_with_buf) - M) < 1.0e-5 * norm(M)
        # allocate_buffer dispatch
        @test allocate_buffer(PartialACA(), Float64) isa HMatrices.ACABuffer{Float64}
        @test allocate_buffer(TSVD(), Float64) === nothing
        @test all(c -> c isa HMatrices.AbstractCompressor, (PartialACA(), TSVD(), comp_buf))
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
    @testset "blocks below the tolerance" begin
        # every singular value is below the tolerance: truncate to rank zero
        small = 1.0e-12 * M
        R = RkMatrix(1.0e-12 * rand(T, m, 3), rand(T, n, 3))
        for comp in (TSVD(; atol = 1.0e-6), PartialACA(; atol = 1.0e-6, recompress = true))
            @test rank(comp(small)) == 0
            @test rank(comp(zero(M))) == 0
        end
        @test rank(HMatrices.compress!(copy(R), TSVD(; atol = 1.0e-6))) == 0
        # rank-zero inputs are left as they are
        R0 = RkMatrix(zeros(T, m, 0), zeros(T, n, 0))
        @test rank(HMatrices.compress!(R0, TSVD(; atol = 1.0e-6))) == 0
    end
    @testset "single precision" begin
        M32 = ComplexF32.(M)
        # the default tolerance follows the precision of the entries
        for comp in (PartialACA(), TSVD(), PartialACA(; recompress = true))
            R = comp(M32)
            @test eltype(R) == ComplexF32
            @test rank(R) < 20
            @test norm(Matrix(R) - M32) < 1.0e-3 * norm(M32)
        end
        # a tolerance that cannot be reached is an error, not a full-rank block
        for comp in (PartialACA(; rtol = 1.0e-8), TSVD(; rtol = 1.0e-8), PartialACA(; rtol = 1.0e-8, recompress = true))
            @test_throws ArgumentError comp(M32)
            @test rank(comp(M)) < m
        end
        @test_throws ArgumentError PartialACA(; rtol = 1.0e-16)(M)
        # an atol below the precision stops at the precision instead of at full rank,
        # also with an explicit `rtol = 0` or `rtol = nothing`
        makers = (kw -> PartialACA(; kw...), kw -> TSVD(; kw...), kw -> PartialACA(; kw..., recompress = true))
        for rtol in (0, nothing), comp in makers
            R = comp((; atol = 1.0e-12, rtol))(M32)
            @test rank(R) < 20
            @test norm(Matrix(R) - M32) < 1.0e-5 * norm(M32)
            # with a rank only, the rest of the block is below the precision
            @test rank(comp((; rank = 50, rtol))(M32)) < 20
        end
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
        @test new_norm ≈ HMatrices._update_frob_norm(old_norm, A, B)
        # the smallest singular value of rectangular entries
        @test HMatrices.min_svd_vals(SMatrix{3, 2}(1.0, 0, 0, 0, 2, 0)) ≈ 1
        @test HMatrices.min_svd_vals(SMatrix{2, 3}(1.0, 0, 0, 2, 0, 0)) ≈ 1
        # recompression of matrix entries, through the equivalent scalar matrix
        fro(X) = sqrt(sum(x -> norm(x)^2, X))
        for rtol in (1.0e-4, 1.0e-8)
            Ra = PartialACA(; rtol)(M)
            Rw = PartialACA(; rtol, recompress = true)(M)
            Rt = TSVD(; rtol)(M)
            @test rank(Rw) < rank(Ra) && rank(Rt) == rank(Rw)
            @test fro(Matrix(Rw) - M) < 5 * rtol * fro(M)
            @test fro(Matrix(Rt) - M) < 5 * rtol * fro(M)
        end
    end
end
