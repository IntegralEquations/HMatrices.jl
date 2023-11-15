module HMatrices

const PROJECT_ROOT = pkgdir(HMatrices)

using StaticArrays
using LinearAlgebra
using Statistics: median
using Printf
using RecipesBase
using Distributed
using Base.Threads
using AbstractTrees
using DataFlowTasks

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

# interface of DataFlowTasks
function DataFlowTasks.memory_overlap(H1::HMatrix, H2::HMatrix)
    isempty(intersect(rowrange(H1), rowrange(H2))) && (return false)
    isempty(intersect(colrange(H1), colrange(H2))) && (return false)
    return true
end
function DataFlowTasks.memory_overlap(H1::HMatrix, pairs::Vector{<:NTuple{2,<:HMatrix}})
    for (A, B) in pairs
        DataFlowTasks.memory_overlap(H1, A) && (return true)
        DataFlowTasks.memory_overlap(H1, B) && (return true)
    end
    return false
end
function DataFlowTasks.memory_overlap(pairs::Vector{<:NTuple{2,<:HMatrix}}, H::HMatrix)
    return DataFlowTasks.memory_overlap(H, pairs)
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
    assemble_hmatrix,
    # macros
    @hprofile

end
