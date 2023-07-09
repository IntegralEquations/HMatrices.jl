using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays
using HMatrices: RkMatrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

N, r = 500, 3
atol = 1e-6
X = Y = rand(SVector{3,Float64}, N)
splitter = CardinalitySplitter(; nmax=50)
Xclt = Yclt = ClusterTree(X, splitter)
adm = StrongAdmissibilityStd(; eta=3)
rtol = 1e-5
comp = PartialACA(; rtol=rtol)
K = helmholtz_matrix(X, Y, 1.3)
H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads=false, distributed=false)
T = eltype(H)
_H = Matrix(H; global_index=false)
R = RkMatrix(rand(T, N, r), rand(T, N, r))
_R = Matrix(R)
M = rand(T, N, N)
_M = Matrix(M)
a = rand()

@testset "axpy!" begin
    @test axpy!(a, M, deepcopy(R)) ≈ a * _M + _R
    tmp = axpy!(a, M, deepcopy(H))
    @test Matrix(tmp; global_index=false) ≈ a * _M + _H
    @test axpy!(a, R, deepcopy(M)) ≈ a * _R + _M
    @test axpy!(a, R, deepcopy(R)) ≈ a * _R + _R
    tmp = axpy!(a, R, deepcopy(H))
    Matrix(tmp; global_index=false) ≈ a * _R + _H
    @test axpy!(a, H, deepcopy(M)) ≈ a * _H + _M
    @test axpy!(a, H, deepcopy(R)) ≈ a * _H + _R
    tmp = axpy!(a, H, deepcopy(H))
    @test Matrix(tmp; global_index=false) ≈ a * _H + _H
end
