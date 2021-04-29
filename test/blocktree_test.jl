using HMatrices
using StaticArrays
using Test

using HMatrices: StrongAdmissibilityStd, ClusterTree, BlockTree, isadmissible

@testset "BlockClusterTree" begin
    Xdata    = rand(SVector{2,Float64},1000)
    adm_fun  = StrongAdmissibilityStd(2)
    Xclt     = ClusterTree(data=Xdata,reorder=false)
    bclt     = BlockTree(Xclt,Xclt,adm_fun)
    @test isadmissible(bclt) == false
    adm_fun  = StrongAdmissibilityStd(4)
    Ydata    = map(x -> x + SVector(10,0),Xdata)
    Yclt  = ClusterTree(data=Ydata,reorder=false)
    bclt = BlockTree(Xclt,Yclt,adm_fun)
    @test isadmissible(bclt) == true
end
