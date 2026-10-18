using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

using HMatrices: RkMatrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

Random.seed!(1)

m = 2000
n = 2000

X = rand(SVector{3, Float64}, m)
Y = [rand(SVector{3, Float64}) for _ in 1:n]
splitter = CardinalitySplitter(; nmax = 50)
Xclt = ClusterTree(X, splitter)
Yclt = ClusterTree(Y, splitter)
adm = StrongAdmissibilityStd(; eta = 3)
atol = 1.0e-5
comp = PartialACA(; atol)
K = laplace_matrix(X, Y)
H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads = false)

H_full = Matrix(H; global_index = false)

R = HMatrices.RkMatrix(rand(m, 10), rand(n, 10))
P = HMatrices.RkMatrix(rand(m, 10), rand(n, 10))

α = rand() - 0.5
β = rand() - 0.5

@testset "hmul!" begin
    C = deepcopy(H)
    tmp = β * H_full + α * H_full * H_full
    HMatrices.hmul!(C, H, H, α, β, PartialACA(; atol = 1.0e-6))
    @test Matrix(C; global_index = false) ≈ tmp
    # adjoint
    C = deepcopy(H)
    tmp = β * H_full + α * adjoint(H_full) * H_full
    HMatrices.hmul!(C, adjoint(H), H, α, β, PartialACA(; atol = 1.0e-6))
    @test Matrix(C; global_index = false) ≈ tmp
    # a compressor with its own buffer type
    C = deepcopy(H)
    tmp = β * H_full + α * H_full * H_full
    HMatrices.hmul!(C, H, H, α, β, ACAWithRecompression(PartialACA(; atol = 1.0e-6)))
    @test Matrix(C; global_index = false) ≈ tmp
end

@testset "hmul! with adjoints" begin
    # kernel without the large regularized diagonal, which would hide errors in
    # the other entries of the products
    Xs = X[1:800]
    Xsclt = ClusterTree(Xs, CardinalitySplitter(; nmax = 30))
    f = (x, y) -> x == y ? 0.0im : exp(2im * norm(x - y)) / (4π * norm(x - y))
    Hs = assemble_hmatrix(KernelMatrix(f, Xs, Xs), Xsclt, Xsclt; comp = PartialACA(; atol = 1.0e-10), threads = false)
    Ms = Matrix(Hs; global_index = false)
    for opA in (identity, adjoint), opB in (identity, adjoint)
        C = deepcopy(Hs)
        HMatrices.hmul!(C, opA(Hs), opB(Hs), -1, 1, PartialACA(; atol = 1.0e-10))
        exact = Ms - opA(Ms) * opB(Ms)
        @test norm(Matrix(C; global_index = false) - exact) < 1.0e-8 * norm(exact)
    end
end

@testset "hmul! with zero rows" begin
    # compactly supported kernel: rows of the products sampled by the compressor
    # are zero up to rounding errors, and must not be used as pivots
    Xc = X[1:1000]
    Xcclt = ClusterTree(Xc, CardinalitySplitter(; nmax = 30))
    f = (x, y) -> (d = norm(x - y); x == y ? 10.0 : d < 0.3 ? exp(-d) / (4π * d) : 0.0)
    for comp in (PartialACA(; atol = 1.0e-8), ACAWithRecompression(PartialACA(; atol = 1.0e-8)))
        Hc = assemble_hmatrix(KernelMatrix(f, Xc, Xc), Xcclt, Xcclt; comp, threads = false)
        Mc = Matrix(Hc; global_index = false)
        C = deepcopy(Hc)
        HMatrices.hmul!(C, Hc, Hc, -1, 1, comp)
        exact = Mc - Mc * Mc
        # the sharp cutoff of the kernel limits the accuracy of the partial ACA
        @test norm(Matrix(C; global_index = false) - exact) < 1.0e-2 * norm(exact)
    end
end

@testset "gemv" begin
    T = eltype(H)
    m, n = size(H)
    x = rand(T, n)
    y = rand(T, m)

    @testset "serial" begin
        exact = β * y + α * H_full * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * Matrix(H; global_index = true) * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = true)
        @test exact ≈ approx

        # multiply by adjoint
        adjH = adjoint(deepcopy(H))
        exact = β * y + α * adjoint(H_full) * x
        approx = mul!(copy(y), adjH, x, α, β; threads = false, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * adjoint(Matrix(H; global_index = true)) * x
        approx = mul!(copy(y), adjH, x, α, β; threads = false, global_index = true)
        @test exact ≈ approx
    end

    @testset "threads" begin
        exact = β * y + α * H_full * x
        approx = mul!(copy(y), H, x, α, β; threads = true, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * Matrix(H; global_index = true) * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = true)
        @test exact ≈ approx

        # multiply by adjoint
        adjH = adjoint(deepcopy(H))
        exact = β * y + α * adjoint(H_full) * x
        approx = mul!(copy(y), adjH, x, α, β; threads = true, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * adjoint(Matrix(H; global_index = true)) * x
        approx = mul!(copy(y), adjH, x, α, β; threads = true, global_index = true)
        @test exact ≈ approx
    end

    @testset "hermitian" begin
        threads = false
        for threads in (true, false)
            K = laplace_matrix(X, X)
            Hsym = assemble_hmatrix(Hermitian(K), Xclt, Xclt; adm, comp, threads)
            H = assemble_hmatrix(K, Xclt, Xclt; adm, comp, threads)
            x = rand(n)
            y1 = mul!(zero(x), H, x, 1, 0; threads)
            y2 = mul!(zero(x), Hsym, x, 1, 0; threads)
            @test y1 ≈ y2
        end
    end
end

@testset "gemm" begin
    α = rand() - 0.5
    β = rand() - 0.5
    T = eltype(H)
    m, n = size(H)
    k = 10
    x = rand(T, n, k)
    y = rand(T, m, k)

    @testset "serial" begin
        exact = β * y + α * H_full * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * Matrix(H; global_index = true) * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = true)
        @test exact ≈ approx
    end

    @testset "threads" begin
        exact = β * y + α * H_full * x
        approx = mul!(copy(y), H, x, α, β; threads = true, global_index = false)
        @test exact ≈ approx
        exact = β * y + α * Matrix(H; global_index = true) * x
        approx = mul!(copy(y), H, x, α, β; threads = false, global_index = true)
        @test exact ≈ approx
    end

    @test_throws DimensionMismatch mul!(zeros(T, m, 2), H, rand(T, n, 3))
end
