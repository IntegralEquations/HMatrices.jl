SUITE["Assembly"]         = BenchmarkGroup(["assembly","hmatrix"])

using ComputationalResources

# parameters
N    = 50_000
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
Xclt      = HMatrices.ClusterTree(X,splitter,reorder)
adm       = HMatrices.StrongAdmissibilityStd(eta)
bclt      = HMatrices.BlockTree(Xclt,Xclt,adm)

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
resource  = CPU1()
comp      = HMatrices.PartialACA(rtol=rtol)

SUITE["Assembly"]["Laplace kernel $N"] = @benchmarkable $HMatrix($K,$bclt,$comp)
