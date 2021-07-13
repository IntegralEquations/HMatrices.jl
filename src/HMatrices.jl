module HMatrices

using StaticArrays: sort
using StaticArrays
using LinearAlgebra
using Statistics: median
using ComputationalResources
using LoopVectorization
using TimerOutputs
using Printf
using RecipesBase
using WavePropBase

WavePropBase.@import_interface
using WavePropBase.Geometry
using WavePropBase.Utils

import AbstractTrees

include("utils.jl")
include("hilbertcurve.jl")
include("kernelmatrix.jl")
include("lowrankmatrices.jl")
include("compressor.jl")
include("hmatrix.jl")

export
    # types (re-exported)
    ClusterTree,
    CardinalitySplitter,
    DyadicSplitter,
    GeometricSplitter,
    GeometricMinimalSplitter,
    HyperRectangle,
    # types
    HMatrix

end
