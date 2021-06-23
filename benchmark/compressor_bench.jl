SUITE["Compressors"]         = BenchmarkGroup(["aca","compression","rkmatrix"])

rtol               = 1e-6
N                  = 1000
X                  = rand(SVector{3,Float64},N)
Y                  = map(x->x+SVector(3,3,3),X)
K                  = HMatrices.LaplaceMatrix(X,Y)
irange             = 1:N
jrange             = 1:N

compressors   = [PartialACA(rtol=rtol),ACA(rtol=rtol),TSVD(rtol=rtol)]

for comp in compressors
    SUITE["Compressors"][string(comp)] = @benchmarkable $comp($K,$irange,$jrange) seconds=0.1
end
