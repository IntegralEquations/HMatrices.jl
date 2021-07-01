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

import AbstractTrees

include("utils.jl")
include("hilbertcurve.jl")
include("kernelmatrix.jl")
include("hyperrectangle.jl")
include("clustertree.jl")
include("lowrankmatrices.jl")
include("compressor.jl")
include("hmatrix.jl")
include("plotrecipes.jl")

export HMatrix

end
