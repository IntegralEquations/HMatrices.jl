using Test
using StaticArrays
using HMatrices
using HMatrices: ClusterTree, CardinalitySplitter, StrongAdmissibilityStd, PartialACA
using LinearAlgebra
using SparseArrays
using Random

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

# a compressor that only implements the method with index ranges, without annotations
struct RangeOnlyCompressor <: HMatrices.AbstractCompressor
    rtol::Float64
end
function (c::RangeOnlyCompressor)(K, irange, jrange, buf)
    return HMatrices.PartialACA(; rtol = c.rtol)(K, irange, jrange)
end

# a matrix that can only be evaluated by blocks of indices
struct BlockOnlyMatrix{M} <: AbstractMatrix{Float64}
    M::M
end
Base.size(B::BlockOnlyMatrix) = size(B.M)
Base.getindex(::BlockOnlyMatrix, i::Int, j::Int) = error("evaluated entry by entry")
function HMatrices.getblock!(out, B::BlockOnlyMatrix, I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer})
    for (jl, j) in enumerate(J), (il, i) in enumerate(I)
        out[il, jl] = B.M[i, j]
    end
    return out
end

@testset "Assemble" begin
    m, n = 1_000, 1_000
    X = Y = rand(SVector{3, Float64}, m)
    splitter = CardinalitySplitter(; nmax = 20)
    Xclt = Yclt = ClusterTree(X, splitter)
    adm = StrongAdmissibilityStd(; eta = 3)
    rtol = 1.0e-5
    comp = PartialACA(; rtol = rtol)
    # Laplace
    for threads in (true, false)
        K = laplace_matrix(X, Y)
        H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads)
        @test norm(Matrix(K) - Matrix(H)) < rtol * norm(Matrix(K))
        H = assemble_hmatrix(K; threads)
        @test norm(Matrix(K) - Matrix(H)) < rtol * norm(Matrix(K))
        adjH = adjoint(H)
        H_full = Matrix(H)
        adjH_full = adjoint(H_full)

        @testset "getcol" begin
            Hloc = HMatrices._tree_matrix(H)
            @test Hloc[:, 666] ≈ HMatrices.getcol(H, 666)
            @test adjoint(Hloc)[:, 666] ≈ HMatrices.getcol(adjH, 666)
        end
        # a kernel with a vectorized `getblock!`, called with permuted indices
        Kv = LaplaceMatrixVec(X, Y)
        I, J = rand(1:m, 7), rand(1:n, 5)
        @test HMatrices.getblock!(zeros(7, 5), Kv, I, J) ≈ Matrix(K)[I, J] rtol = 1.0e-12
        P = HMatrices.PermutedMatrix(Kv, HMatrices.rowperm(H), HMatrices.colperm(H))
        Mp = Matrix(K)[HMatrices.rowperm(H), HMatrices.colperm(H)]
        @test HMatrices.getblock!(zeros(m), P, 1:m, 3) ≈ Mp[:, 3] rtol = 1.0e-12 # an ACA column
        @test HMatrices.getblock!(zeros(n), adjoint(P), 1:n, 4) ≈ Mp[4, :] rtol = 1.0e-12 # an ACA row
        Hv = assemble_hmatrix(Kv, Xclt, Yclt; adm, comp, threads)
        @test norm(Matrix(K) - Matrix(Hv)) < rtol * norm(Matrix(K))
        # Elastostatic
        K = elastostatic_matrix(X, Y, 1.1, 1.2)
        H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads)
        @test norm(Matrix(K) - Matrix(H)) < rtol * norm(Matrix(K))
    end
end

@testset "Band sparse arrays" begin
    m = 1000
    n = 1000

    X = rand(SVector{3, Float64}, m)
    Y = X
    splitter = CardinalitySplitter(; nmax = 40)
    Xclt = ClusterTree(X, splitter)
    Yclt = ClusterTree(Y, splitter)
    adm = StrongAdmissibilityStd(; eta = 3)
    rtol = 1.0e-5
    comp = PartialACA(; rtol = rtol)
    @testset "Scalar problem (laplace)" begin
        K = laplace_matrix(X, Y)
        H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads = false)
        H_full = Matrix(H)
        T = eltype(H)
        m, n = size(H)
        S = spdiagm(0 => rand(T, n))
        Hnew = axpy!(true, S, deepcopy(H))
        @test Matrix(Hnew) == (H_full + Matrix(S))
    end
    @testset "Vector problem (elasticity)" begin
        K = elastostatic_matrix(X, Y, 1, 1)
        H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads = false)
        H_full = Matrix(H)
        T = eltype(H)
        m, n = size(H)
        S = spdiagm(0 => rand(T, n))
        Hnew = axpy!(true, S, deepcopy(H))
        @test Matrix(Hnew) == (H_full + Matrix(S))
    end
end

@testset "Precision of adding any sparse matrix to an H-matrix with regard to the matrix - vector multiplication" begin
    m = 1000
    n = 1000

    X = rand(SVector{3, Float64}, m)
    Y = X
    splitter = CardinalitySplitter(; nmax = 40)
    Xclt = ClusterTree(X, splitter)
    Yclt = ClusterTree(Y, splitter)
    adm = StrongAdmissibilityStd(; eta = 3)
    rtol = 1.0e-5
    comp = PartialACA(; rtol = rtol)
    @testset "Scalar problem (laplace)" begin
        K = laplace_matrix(X, Y)
        H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads = false)
        H_full = Matrix(H)
        T = eltype(H)
        m, n = size(H)
        S = sprand(T, m, n, 0.01)
        # compare the increments: the large diagonal of `H` would hide errors
        for a in (true, 2) # the scaling also applies to admissible blocks
            Hnew = axpy!(a, S, deepcopy(H))
            @test norm(Matrix(Hnew) - H_full - a * Matrix(S)) < 1.0e-8 * norm(a * S)
        end
    end
    @testset "Vector problem (elasticity)" begin
        K = elastostatic_matrix(X, Y, 1, 1)
        H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads = false)
        H_full = Matrix(H)
        T = eltype(H)
        m, n = size(H)
        S = sprand(T, m, n, 0.1)
        # compare the increments: the large diagonal of `H` would hide errors
        for a in (true, 2) # the scaling also applies to admissible blocks
            Hnew = axpy!(a, S, deepcopy(H))
            @test norm(Matrix(Hnew) - H_full - a * Matrix(S)) < 1.0e-8 * norm(a * S)
        end
    end
end

@testset "Assemble from points" begin
    # a matrix indexed by (i, j), with points used only to cluster its indices
    S = single_layer_sphere(1500)
    X = S.X
    f = S.f
    K = KernelMatrix((i, j) -> f(X[i], X[j]), 1:1500, 1:1500)
    M = Matrix(K)
    X0 = copy(X)
    H = assemble_hmatrix(K, X; rtol = 1.0e-8)
    @test X == X0 # the points are not permuted in place
    @test norm(Matrix(H) - M) < 1.0e-7 * norm(M)
    # weak admissibility and leaf size
    Hw = assemble_hmatrix(K, X; rtol = 1.0e-8, eta = Inf, nmax = 20, threads = false)
    @test norm(Matrix(Hw) - M) < 1.0e-7 * norm(M)
    @test maximum(length, HMatrices.leaves(HMatrices.rowtree(Hw))) <= 20
    # some compressed blocks are too close for the strong admissibility
    strong = StrongAdmissibilityStd(; eta = 3)
    @test any(HMatrices.leaves(Hw)) do l
        return HMatrices.isadmissible(l) && !strong(HMatrices.rowtree(l), HMatrices.coltree(l))
    end
    # Hermitian matrices give a Hermitian HMatrix, ready for `cholesky`
    Hh = assemble_hmatrix(Hermitian(K), X; rtol = 1.0e-8)
    @test Hh isa HMatrices.HermitianHMatrix
    y = rand(1500)
    @test norm(cholesky(Hh; rtol = 1.0e-8) \ y - M \ y) < 1.0e-6 * norm(M \ y)
    # the product in place of its argument
    z = copy(y)
    @test norm(mul!(z, H, z) - M * y) < 1.0e-7 * norm(M * y)
    @test_throws ArgumentError assemble_hmatrix(Hermitian(K), X, copy(X))
    # `getblock!` of the matrix is used for every block, with vectors of indices
    Hb = assemble_hmatrix(BlockOnlyMatrix(M), X; rtol = 1.0e-8, threads = false)
    @test norm(Matrix(Hb) - M) < 1.0e-7 * norm(M)
    # integer coordinates
    P = [SVector(i % 50, i ÷ 50) for i in 0:1499]
    Kint = KernelMatrix((i, j) -> exp(-norm(P[i] - P[j]) / 10), 1:1500, 1:1500)
    @test norm(Matrix(assemble_hmatrix(Kint, P; rtol = 1.0e-8)) - Matrix(Kint)) < 1.0e-7 * norm(Matrix(Kint))
    # a kernel whose element type is not concrete
    @test_throws ArgumentError assemble_hmatrix(KernelMatrix((i, j) -> rand() > 2 ? 1 : 1.0, 1:1500, 1:1500), X)
    # a compactly supported kernel has exactly zero blocks, which are not ACA failures
    Kc = KernelMatrix((x, y) -> norm(x - y) < 0.2 ? f(x, y) : 0.0, X, X)
    @test_logs assemble_hmatrix(Kc; rtol = 1.0e-8, threads = false)
    # a zero block of a dense matrix is not an ACA failure
    Md = Matrix(Kc)
    @test_logs assemble_hmatrix(Md, X; rtol = 1.0e-8, threads = false)
    # the number of points must match the size of the matrix
    @test_throws DimensionMismatch assemble_hmatrix(K, X[1:1000])
    # in single precision, an atol below the precision stops at the precision
    K32 = KernelMatrix((i, j) -> Float32(f(X[i], X[j])), 1:1500, 1:1500)
    H32 = assemble_hmatrix(K32, X; atol = 1.0e-12, threads = false)
    @test HMatrices.compression_ratio(H32) > 1
    @test norm(Matrix(H32) - Matrix(K32)) < 1.0e-5 * norm(Matrix(K32))
    # a compressor that only defines the method with index ranges
    Hr = assemble_hmatrix(K, HMatrices.ClusterTree(X), HMatrices.ClusterTree(X); comp = RangeOnlyCompressor(1.0e-8), threads = false)
    @test norm(Matrix(Hr) - M) < 1.0e-7 * norm(M)
    # the same through a kernel matrix on the points
    Kp = KernelMatrix(f, X, X)
    @test assemble_hmatrix(Hermitian(Kp); rtol = 1.0e-8) isa HMatrices.HermitianHMatrix
    @test norm(Matrix(assemble_hmatrix(Kp; rtol = 1.0e-8)) - M) < 1.0e-7 * norm(M)
end

@testset "Rectangular, from points" begin
    # different points for the rows and the columns, in the user ordering
    S = single_layer_sphere(1200)
    X = S.X
    Y = [2 * x for x in S.X[1:800]]
    f = S.f
    K = KernelMatrix((i, j) -> f(X[i], Y[j]), 1:1200, 1:800)
    M = Matrix(K)
    H = assemble_hmatrix(K, X, Y; rtol = 1.0e-8)
    @test norm(Matrix(H) - M) < 1.0e-7 * norm(M)
    x, y = rand(800), rand(1200)
    @test norm(H * x - M * x) < 1.0e-7 * norm(M * x)
    @test norm(H' * y - M' * y) < 1.0e-7 * norm(M' * y)
    # as for dense matrices, the identity can only be added to a square matrix
    @test_throws DimensionMismatch H + 2I
end

@testset "Identity with different row and column orderings" begin
    n = 1200
    S = single_layer_sphere(n)
    X = S.X
    Y = X[randperm(n)]
    K = KernelMatrix(S.f, X, Y)
    M = Matrix(K)
    H = assemble_hmatrix(K, X, Y; rtol = 1.0e-10, threads = false)
    @test HMatrices.rowperm(H) != HMatrices.colperm(H)
    @test norm(Matrix(H + 2I) - (M + 2I)) < 1.0e-8 * norm(M)
    @test norm(Matrix(axpy!(3, I, deepcopy(H))) - (M + 3I)) < 1.0e-8 * norm(M)
    # the factors of a product must be clustered consistently
    @test_throws ArgumentError HMatrices.hmul!(deepcopy(H), H, H, 1, 0, HMatrices.PartialACA())
    # operations in the user ordering are not defined on a block
    blk = HMatrices.children(H)[1, 1]
    @test_throws ArgumentError Matrix(blk)
    @test_throws ArgumentError mul!(zeros(size(blk, 1)), blk, rand(size(blk, 2)))
    # a scalar of a wider type than the vector
    x = rand(n)
    @test mul!(zeros(ComplexF64, n), H, x, 2im, 0; threads = false) ≈ 2im * (M * x)
end
