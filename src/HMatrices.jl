module HMatrices

const PROJECT_ROOT = pkgdir(HMatrices)

using StaticArrays
using LinearAlgebra
using Statistics: median, mean
using Printf
using RecipesBase
using Distributed
using Base.Threads
using SparseArrays

const AdjOrMat = Union{Matrix,Adjoint{<:Any,<:Matrix}}

"""
    getblock!(block,K,irange,jrange)

Fill `block` with `K[i,j]` for `i ∈ irange`, `j ∈ jrange`, where `block` is of
size `length(irange) × length(jrange)`.

A default implementation exists which relies on `getindex(K,i,j)`, but this
method can be overloaded for better performance if e.g. a vectorized way of
computing a block is available.
"""
function getblock!(out, K, irange_, jrange_)
    irange = irange_ isa Colon ? axes(K, 1) : irange_
    jrange = jrange_ isa Colon ? axes(K, 2) : jrange_
    for (jloc, j) in enumerate(jrange)
        for (iloc, i) in enumerate(irange)
            out[iloc, jloc] = K[i, j]
        end
    end
    return out
end

function getblock!(out, Kadj::Adjoint, irange_, j::Int)
    getblock!(transpose(out), parent(Kadj), j:j, irange_)
    return out .= conj.(out)
end

"""
    use_threads()::Bool

Default choice of whether threads will be used throughout the package.
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
include("multiplication.jl")
include("triangular.jl")
include("lu.jl")
include("cholesky.jl")

if !isdefined(Base, :get_extension) # for julia version < 1.9
    include("../ext/HBEAST/HBEAST.jl")
end

export ClusterTree,
    CardinalitySplitter,
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
    TSVD,
    # functions
    compression_ratio,
    assemble_hmatrix

end
