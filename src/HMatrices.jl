module HMatrices

const PROJECT_ROOT = pkgdir(HMatrices)

using TimerOutputs
using StaticArrays
using LinearAlgebra
using Statistics: median
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


"""
    get_block!(block,K,irange,jrange,[append=false])

Fill `block` with `K[irange,jrange]`. If `append` is `true`, the data is added
to the current values of `block`; otherwise `block` is overwritten.
"""
function get_block!(out,K,irange_,jrange_)
    irange = irange_ isa Colon ? axes(K,1) : irange_
    jrange = jrange_ isa Colon ? axes(K,2) : jrange_
    for (jloc,j) in enumerate(jrange)
        for (iloc,i) in enumerate(irange)
            out[iloc,jloc] = K[i,j]
        end
    end
    return out
end

function get_block!(out, Kadj::Adjoint, irange_, jrange_)
    get_block!(transpose(out), parent(Kadj), jrange_, irange_)
    out .= conj.(out)
end

function get_block(K, irange, jrange)
    m = irange isa Colon ? size(K,1) : length(irange)
    n = jrange isa Colon ? size(K,2) : length(jrange)
    T = eltype(K)
    out = zeros(T, m, n)
    get_block!(out, K, irange, jrange)
end

# getcol for regular matrices
function getcol!(col, M::Matrix, j)
    @assert length(col) == size(M, 1)
    return copyto!(col, view(M, :, j))
end
function getcol!(col, adjM::Adjoint{<:Any,<:Matrix}, j)
    @assert length(col) == size(adjM, 1)
    return copyto!(col, view(adjM, :, j))
end

getcol(M::Matrix, j) = M[:, j]
getcol(adjM::Adjoint{<:Any,<:Matrix}, j) = adjM[:, j]

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
    assemble_hmatrix
end
