using LinearAlgebra
using LowRankApprox
using HMatrices
using BenchmarkTools

n = 1024
A = matrixlib(:hilb, n, n)

comp = PartialACA(rtol=1e-6)
comp2 = TSVD(rtol=1e-6)

R  = comp(A)
R2 = comp2(A)

norm(A - Matrix(R))

opts = LRAOptions(rtol=1e-6,sketch=:sub)

F = pqrfact(A, opts)
V = idfact(A, opts)
S = psvdfact(A, opts)
C = curfact(A, opts)

@btime $(comp)($A);
@btime $(comp2)($A);
@btime pqrfact($A,$opts);
@btime idfact($A,$opts);
@btime psvd($A,$opts);
@btime curfact($A,$opts);


m,n  = 1000,1000
X    = rand(SVector{3,Float64},m)
Y    = [rand(SVector{3,Float64}) .+ 5 for _  in 1:n]
splitter  = CardinalitySplitter(nmax=50)
Xclt      = ClusterTree(X,splitter)
Yclt      = ClusterTree(Y,splitter)
adm       = StrongAdmissibilityStd(eta=3)
rtol      = 1e-5
comp      = PartialACA(rtol=rtol)
# Laplace
A         = LaplaceMatrix(X,Y) |> Matrix

@info rank(comp(A))
@btime $(comp)($A);
@info rank(comp2(A))
@btime $(comp2)($A);
@btime pqrfact($A,$opts);
@btime idfact($A,$opts);
@btime psvdfact($A,$opts);
@btime curfact($A,$opts);

F = psvdfact(A,opts);
