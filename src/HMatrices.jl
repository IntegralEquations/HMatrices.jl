module HMatrices

using StaticArrays
using LinearAlgebra
using Statistics: median
using TimerOutputs
using Printf
using RecipesBase
using WavePropBase
using WavePropBase.Trees
using WavePropBase.Utils
using WavePropBase.Geometry

import AbstractTrees

include("utils.jl")
include("rkmatrix.jl")
include("compressor.jl")
include("hmatrix.jl")
include("addition.jl")
include("multiplication.jl")
include("triangular.jl")
include("lu.jl")

export
    # modules (re-exported)
    Utils,
    Geometry,
    # types (re-exported)
    ClusterTree,
    CardinalitySplitter,
    DyadicSplitter,
    GeometricSplitter,
    GeometricMinimalSplitter,
    HyperRectangle,
    # types
    HMatrix,
    StrongAdmissibilityStd,
    WeakAdmissibilityStd,
    PartialACA,
    ACA,
    TSVD,
    # functions
    compression_ratio,
    print_tree,
    # macros
    @hprofile

end
