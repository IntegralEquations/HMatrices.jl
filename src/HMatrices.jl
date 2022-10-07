module HMatrices

const PROJECT_ROOT = pkgdir(HMatrices)

using StaticArrays
using LinearAlgebra
using Statistics: median
using TimerOutputs
using Printf
using RecipesBase
using Distributed

import WavePropBase:
                     ClusterTree,
                     CardinalitySplitter,
                     DyadicSplitter,
                     GeometricSplitter,
                     GeometricMinimalSplitter,
                     HyperRectangle,
                     AbstractTree,
                     filter_tree,
                     isleaf,
                     children,
                     parent,
                     index_range,
                     container,
                     center,
                     diameter,
                     distance,
                     root_elements,
                     loc2glob

using AbstractTrees: AbstractTrees
import LinearAlgebra: mul!, lu!, lu, LU, ldiv!, rdiv!, axpy!, rank, rmul!, lmul!
import Base: Matrix, adjoint

"""
    const ALLOW_GETINDEX

If set to false, the `getindex(H,i,j)` method will throw an error on
`AbstractHMatrix`.
"""
const ALLOW_GETINDEX = Ref(true)

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
      assemble_hmat,
# macros
      @hprofile

end
