module HMatrices

using StaticArrays
using LinearAlgebra
using Statistics: median

import AbstractTrees

include("utils.jl")
include("hyperrectangle.jl")
include("clustertree.jl")
include("blocktree.jl")

end
