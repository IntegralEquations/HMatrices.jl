module HMatrices

const PROJECT_ROOT = pkgdir(HMatrices)

using StaticArrays
using LinearAlgebra
using Statistics: median
using TimerOutputs
using Printf
using RecipesBase
using Distributed
using Base.Threads
using AbstractTrees: print_tree

using AbstractTrees: AbstractTrees
import LinearAlgebra: mul!, lu!, lu, LU, ldiv!, rdiv!, axpy!, rank, rmul!, lmul!
import Base: Matrix, adjoint, parent

"""
    const ALLOW_GETINDEX

If set to false (default), the `getindex(H,i,j)` method will throw an error on
[`AbstractHMatrix`](@ref) and [`RkMatrix`](@ref).
"""
const ALLOW_GETINDEX = Ref(false)

"""
    use_threads()::Bool

Default choice of whether threads will be used or not throughout the package.
"""
use_threads() = true

"""
    use_global_index()::Bool

Default choice of whether operations will use the global indexing system
throughout the package.
"""
use_global_index() = true

include("utils.jl")
include("hyperrectangle.jl")
include("clustertree.jl")
include("splitter.jl")
include("kernelmatrix.jl")
include("rkmatrix.jl")
include("compressor.jl")
include("hmatrix.jl")
include("dhmatrix.jl")
include("addition.jl")
include("partitions.jl")
include("multiplication.jl")
include("triangular.jl")
include("lu.jl")

export
    # types (re-exported)
    CardinalitySplitter,
    ClusterTree,
    DyadicSplitter,
    GeometricSplitter,
    GeometricMinimalSplitter,
    HyperRectangle,
    # abstract types
    AbstractKernelMatrix,
    # types
    HMatrix,
    KernelMatrix,
    StrongAdmissibilityStd,
    WeakAdmissibilityStd,
    PartialACA,
    ACA,
    TSVD,
    # functions
    compression_ratio,
    print_tree,
    assemble_hmatrix,
    # macros
    @hprofile

end
