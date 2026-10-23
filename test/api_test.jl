using HMatrices
using Test

exported = filter(n -> Base.isexported(HMatrices, n), names(HMatrices))
@test Set(exported) ==
    Set([:HMatrices, :assemble_hmatrix, :HMatrix, :KernelMatrix, :AbstractKernelMatrix, :compression_ratio])
public_names = filter(n -> Base.ispublic(HMatrices, n) && !Base.isexported(HMatrices, n), names(HMatrices; all = true))
@test Set(public_names) == Set(
    [
        :ClusterTree, :AbstractSplitter, :GeometricSplitter, :CardinalitySplitter,
        :PrincipalComponentSplitter, :HyperRectangle, :StrongAdmissibilityStd,
        :WeakAdmissibilityStd, :AbstractCompressor, :PartialACA, :TSVD,
        :allocate_buffer, :getblock!, :rowperm, :colperm, :center,
        :rowelements, :colelements,
    ]
)
for n in union(exported, public_names)
    @test isdefined(HMatrices, n)
    n === :HMatrices || @test Base.Docs.hasdoc(HMatrices, n)
end
