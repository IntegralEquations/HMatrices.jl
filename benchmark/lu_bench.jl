using HMatrices
using StaticArrays
using LinearAlgebra
using ComputationalResources
using BenchmarkTools

# SUITE["Assembly"]         = BenchmarkGroup(["assembly","hmatrix"])

# parameters

N    = 20_000
nmax = 100
eta  = 3
radius = 1
rtol   = 1e-5

# function to generate the point cloud
function points_on_cylinder(radius,step,n,shift=SVector(0,0,0))
    result          = Vector{SVector{3,Float64}}(undef,n)
    length          = 2*π*radius
    pointsPerCircle = length/step
    angleStep       = 2*π/pointsPerCircle
    for i=0:n-1
        x = radius * cos(angleStep*i)
        y = radius * sin(angleStep*i)
        z = step*i/pointsPerCircle
        result[i+1] = shift + SVector(x,y,z)
    end
    return result
end

# create block structure
_step     = 1.75*π*radius/sqrt(N)
X         = points_on_cylinder(radius,_step,N)
splitter  = HMatrices.CardinalitySplitter(nmax)
Xclt      = HMatrices.ClusterTree(X,splitter)
adm       = HMatrices.StrongAdmissibilityStd(eta)

# compression method
comp      = HMatrices.PartialACA(;rtol)

# create your abstract matrix
struct LaplaceMatrix{T} <: AbstractMatrix{T}
    X::Vector{SVector{3,Float64}}
    Y::Vector{SVector{3,Float64}}
end
Base.size(K::LaplaceMatrix) = length(K.X), length(K.Y)
function Base.getindex(K::LaplaceMatrix{T},i::Int,j::Int)::T where {T}
    d = norm(K.X[i] - K.Y[j]) + 1e-10
    return inv(4π*d)
end

K         = LaplaceMatrix{Float64}(X,X)
comp      = HMatrices.PartialACA(rtol=rtol)

H = HMatrix(K,Xclt,Xclt,adm,comp;threads=false)

##

compressor = PartialACA(;atol=1e-6)
lu(H,compressor)

# 15s for 10_000 dofs
@elapsed lu(H,compressor)

# @profview lu(H,compressor)
