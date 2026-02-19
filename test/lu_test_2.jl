using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))
Random.seed!(1)
m = 10
T = Float64
X = points_on_sphere(m)
Y = X
K = laplace_matrix(X, X)
splitter = CardinalitySplitter(; nmax = 5)
Xclt = ClusterTree(X, splitter)
Yclt = ClusterTree(Y, splitter)
adm = StrongAdmissibilityStd(3)
comp = PartialACA(; atol = 1.0e-10)
H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads = false, distributed = false)
Hsym = assemble_hmatrix(
    Hermitian(K),
    Xclt,
    Yclt;
    adm,
    comp,
    threads = false,
    distributed = false,
)
Hfull = Matrix(H; global_index = false)

F = lu(Hfull)

chd = map(HMatrices.children(H)) do x
    return Hfull[HMatrices.rowrange(x), HMatrices.colrange(x)]
end

lu!(chd[1, 1])
ldiv!(UnitLowerTriangular(chd[1, 1]), chd[1, 2])
rdiv!(chd[2, 1], UpperTriangular(chd[1, 1]))
mul!(chd[2, 2], chd[2, 1], chd[1, 2], -1, 1)
lu!(chd[2, 2])

res = map(HMatrices.children(H)) do x
    return F.factors[HMatrices.rowrange(x), HMatrices.colrange(x)]
end

@show norm(res[1, 1] - chd[1, 1], Inf)
@show norm(res[1, 2] - chd[1, 2], Inf)
@show norm(res[2, 1] - chd[2, 1], Inf)
@show norm(res[2, 2] - chd[2, 2], Inf)

# lu version
