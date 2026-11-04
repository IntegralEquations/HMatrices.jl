using HMatrices
using LinearAlgebra
using Test
using StaticArrays
using HMatrices: RkMatrix, PartialACA, TSVD

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

@testset "orthogonality and truncation errors" begin
    m, n, r = 60, 40, 8
    M = rand(m, r) * rand(r, n) + 1.0e-9 * rand(m, n)
    # truncation errors of the first `i` columns
    errs(R) = [norm(M - R.A[:, 1:i] * R.B[:, 1:i]') for i in 1:rank(R)]
    @testset "constructors" begin
        R = RkMatrix(rand(m, 2), rand(n, 2))
        @test R.orthogonality == :none && isempty(R.truncation_error)
        R.orthogonality = :AB
        R.truncation_error = [1.0, 0.0]
        @test copy(R).orthogonality == :AB && copy(R).truncation_error == [1.0, 0.0]
        R.A = rand(m, 2) # assigning a factor invalidates the information
        @test R.orthogonality == :none && isempty(R.truncation_error)
    end
    @testset "PartialACA" begin
        R = PartialACA(; atol = 1.0e-6)(M)
        @test R.orthogonality == :none
        @test length(R.truncation_error) == rank(R)
        @test issorted(R.truncation_error; rev = true)
    end
    @testset "SVD forms" begin
        for R in (
                PartialACA(; atol = 1.0e-6, recompress = true)(M),
                HMatrices.RandomizedSVD(; atol = 1.0e-6)(M),
                HMatrices.compress!(copy(M), TSVD(; atol = 1.0e-6)),
                HMatrices.compress!(RkMatrix(copy(M), Matrix(1.0I, n, n)), TSVD(; atol = 1.0e-6)),
            )
            @test R.orthogonality == :AB
            GA, GB = R.A' * R.A, R.B' * R.B
            @test norm(GA - Diagonal(diag(GA))) <= 1.0e-10 * norm(GA)
            @test norm(GB - Diagonal(diag(GB))) <= 1.0e-10 * norm(GB)
            @test length(R.truncation_error) == rank(R)
            # the estimates are the actual errors, up to the error of the input
            @test all(abs.(R.truncation_error - errs(R)) .<= 1.0e-6 .+ 1.0e-6 * errs(R))
        end
    end
    @testset "in-place modifications" begin
        R = PartialACA(; atol = 1.0e-6, recompress = true)(M)
        e = copy(R.truncation_error)
        rmul!(R, -2.0)
        @test R.orthogonality == :AB && R.truncation_error ≈ 2 * e
        HMatrices._modified_factor!(R, :A) # e.g. a triangular solve on `A`
        @test R.orthogonality == :B && isempty(R.truncation_error)
        HMatrices._modified_factor!(R, :A)
        @test R.orthogonality == :B
        HMatrices._modified_factor!(R, :B)
        @test R.orthogonality == :none
    end
end
