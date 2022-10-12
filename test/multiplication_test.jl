using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

using HMatrices: RkMatrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

# Random.seed!(1)

m = 2000
n = 2000

X = rand(SVector{3,Float64}, m)
Y = [rand(SVector{3,Float64}) for _ in 1:n]
splitter = CardinalitySplitter(; nmax=40)
Xclt = ClusterTree(X, splitter)
Yclt = ClusterTree(Y, splitter)
adm = StrongAdmissibilityStd(; eta=3)
rtol = 1e-5
comp = PartialACA(; rtol=rtol)
K = laplace_matrix(X, Y)
H = assemble_hmat(K, Xclt, Yclt; adm, comp, threads=false, distributed=false)

H_full = Matrix(H; global_index=false)

α = rand() - 0.5
β = rand() - 0.5
@testset "hmul!" begin
    C = deepcopy(H)
    tmp = β * H_full + α * H_full * H_full
    root = HMatrices.hmul!(C, H, H, α, β, PartialACA(; atol=1e-6))
    @test norm(Matrix(C; global_index=false) - tmp,Inf) < 1e-5
end

@testset "gemv" begin
    α = rand() - 0.5
    β = rand() - 0.5
    T = eltype(H)
    m, n = size(H)
    x = rand(T, n)
    y = rand(T, m)

    @testset "serial" begin
        exact = β * y + α * H_full * x
        approx = mul!(copy(y), H, x, α, β; threads=false, global_index=false)
        @test exact ≈ approx
        exact = β * y + α * Matrix(H; global_index=true) * x
        approx = mul!(copy(y), H, x, α, β; threads=false, global_index=true)
        @test exact ≈ approx
    end

    @testset "threads" begin
        exact = β * y + α * H_full * x
        approx = mul!(copy(y), H, x, α, β; threads=true, global_index=false)
        @test exact ≈ approx
        exact = β * y + α * Matrix(H; global_index=true) * x
        approx = mul!(copy(y), H, x, α, β; threads=false, global_index=true)
        @test exact ≈ approx
    end
end
