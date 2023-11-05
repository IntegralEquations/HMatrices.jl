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
using AbstractTrees

"""
    const ALLOW_GETINDEX

If set to false (default), the `getindex(H,i,j)` method will throw an error on
[`AbstractHMatrix`](@ref) and [`RkMatrix`](@ref).
"""
const ALLOW_GETINDEX = Ref(false)

"""
    get_block!(block,K,irange,jrange)

Fill `block` with `K[irange,jrange]`.
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

function getblock!(out, Kadj::Adjoint, irange_, jrange_)
    getblock!(transpose(out), parent(Kadj), jrange_, irange_)
    return out .= conj.(out)
end

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
    assemble_hmatrix,
    # macros
    @hprofile

end
