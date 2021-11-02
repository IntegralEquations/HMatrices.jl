using HMatrices
using StaticArrays
using LinearAlgebra
using ComputationalResources
using BenchmarkTools

HMatrices.debug()

# SUITE["Assembly"]         = BenchmarkGroup(["assembly","hmatrix"])

# parameters

N    = 300_000
nmax = 200
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

H = HMatrix(K,Xclt,Xclt,adm,comp)

x = rand(N)
y = similar(x)

# @btime mul!($y,$H,$x);

nt = Threads.nthreads()
hilbert_partition = HMatrices.hilbert_partitioning(H,nt)
row_partition = HMatrices.row_partitioning(H,nt)
col_partition = HMatrices.col_partitioning(H,nt)

BLAS.set_num_threads(1)
@benchmark HMatrices._mul_CPU!($y,$H,$x,1,0)
@benchmark HMatrices._mul_threads!($y,$H,$x,1,0)
@benchmark HMatrices._mul_static!($y,$H,$x,1,0,$hilbert_partition)
@benchmark HMatrices._mul_static!($y,$H,$x,1,0,$row_partition)
@benchmark HMatrices._mul_static!($y,$H,$x,1,0,$col_partition)


# SUITE["Assembly"]["Laplace kernel $N"] = @benchmarkable $HMatrix(,$comp)
