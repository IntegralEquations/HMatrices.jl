################################################################################
## CROSS-CONSTRUCTORS AND CONVERSIONS
################################################################################

# Matrix ==> RkMatrix
function RkMatrix(F::Matrix,tol=0)
    F = LinearAlgebra.svd(F)
    max_rank = findfirst(x -> x<=tol, F.S)
    max_rank === nothing && (max_rank = length(F.S))
    A  = F.U[:,1:max_rank]*LinearAlgebra.Diagonal(F.S[1:max_rank])
    B  = F.V[:,1:max_rank]
    return RkMatrix(A,B)
end

## HMatrix ==> RkMatrix
# function RkMatrix{T}(block::HMatrix,tol) where {T}
#     if isleaf(block)
#         if issparse(block)
#             return block.data
#         else
#             return RkMatrix{T}(block.data,tol)
#         end
#     else
#         rmat11 = RkMatrix{T}(block.children[1],tol)
#         rmat12 = RkMatrix{T}(block.children[2],tol)
#         rmat21 = RkMatrix{T}(block.children[3],tol)
#         rmat22 = RkMatrix{T}(block.children[4],tol)
#         tmp1   = trunc(hcat(rmat11,rmat12),tol)
#         tmp2   = trunc(hcat(rmat21,rmat22),tol)
#         return trunc(vcat(tmp1,tmp2),tol)
#     end
# end
# RkMatrix(block::HMatrix{T},tol) where {T} = RkMatrix{T}(block,tol)

# RkMatrix ==> Matrix
function Base.Matrix(R::RkMatrix{<:Number})
    Matrix(R.A*R.Bt)
end
function Base.Matrix(R::RkMatrix{<:SMatrix})
    # collect must be used when we have a matrix of `SMatrix` because of this:
    # https://github.com/JuliaArrays/StaticArrays.jl/issues/966#issuecomment-943679214
    R.A*collect(R.Bt)
end

# HMatrix ==> Matrix
"""
    Matrix(H::HMatrix;permute=false)

Convert `H` to a `Matrix`. If `permute`, the entries are given in the global
indexing system (see [`HMatrix`](@ref) for more information).
"""
Matrix(hmat::HMatrix;global_index=false) = Matrix{eltype(hmat)}(hmat;global_index)
function Matrix{T}(hmat::HMatrix;global_index) where {T}
    M = zeros(T,size(hmat)...)
    piv = pivot(hmat)
    for block in PreOrderDFS(hmat)
        hasdata(block) || continue
        irange = rowrange(block) .- piv[1] .+ 1
        jrange = colrange(block) .- piv[2] .+ 1
        M[irange,jrange] += Matrix(block.data)
    end
    if global_index
        P = PermutedMatrix(M,invperm(rowperm(hmat)),invperm(colperm(hmat)))
        return Matrix(P)
    else
        return M
    end
end


# RkMatrix ==> HMatrix
# Note: converting RkMatrix to HMatrix requires the HMatrix structure
function Base.fill!(H::HMatrix,R::RkMatrix)
    @assert size(H) === size(R)
    shift = get_pivot(H) .- 1
    for block in get_leaves(H)
        irange = row_range(block) .- shift[1]
        jrange = col_range(block) .- shift[2]
        block.data = RkMatrix(R.A[irange,:],R.Bt[:,jrange],R.tol)
    end
    return H
end
