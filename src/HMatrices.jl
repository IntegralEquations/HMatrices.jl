module HMatrices

using StaticArrays
using LinearAlgebra
using Statistics: median
using ComputationalResources
using LoopVectorization
using TimerOutputs
using Printf
using RecipesBase

using WavePropBase
using WavePropBase.Geometry
using WavePropBase.Utils

WavePropBase.@import_interface

import AbstractTrees

include("utils.jl")
include("hilbertcurve.jl")
include("kernelmatrix.jl")
include("lowrankmatrices.jl")
include("compressor.jl")
include("hmatrix.jl")
include("hgemv.jl")

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
