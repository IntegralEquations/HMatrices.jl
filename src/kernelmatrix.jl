"""
    abstract type AbstractKernelMatrix{T} <: AbstractMatrix{T}

Interface for abstract matrices represented through a function `f`, a target
point cloud `X::Vector{SVector}`, and source point cloud `Y::Vector{SVector}`.
Concrete subtypes should implement at least

    `Base.getindex(K::AbstractKernelMatrix,i::Int,j::Int)`

See [`LaplaceMatrix`](@ref) for an example of an implementation.
"""
abstract type AbstractKernelMatrix{T} <: AbstractMatrix{T} end

Base.size(K::AbstractKernelMatrix) = size(K.X,1),size(K.Y,1)

"""
    struct LaplaceMatrix{T,Td} <: AbstractKernelMatrix{T}

Freespace Greens function for Laplace's equation in three dimensions. The type
parameter `T` is the return type, and `Td` is the type used to reprenset the
points `X` and `Y`.
"""
struct LaplaceMatrix{T,Td} <: AbstractKernelMatrix{T}
    X::Matrix{Td}
    Y::Matrix{Td}
    function LaplaceMatrix{T}(X::Matrix{Td},Y::Matrix{Td}) where {T,Td}
        @assert size(X,2) == size(Y,2) == 3
        new{T,Td}(X,Y)
    end
end

function LaplaceMatrix{T}(_X::Vector{SVector{3,Td}},_Y::Vector{SVector{3,Td}}) where {T,Td}
    X = reshape(reinterpret(Td,_X), 3,:) |> transpose |> collect
    Y = reshape(reinterpret(Td,_Y), 3,:) |> transpose |> collect
    LaplaceMatrix{T}(X,Y)
end
LaplaceMatrix(args...) = LaplaceMatrix{Float64}(args...) # default to Float64

@inline function Base.getindex(K::LaplaceMatrix{T},i::Int,j::Int)::T where {T}
    d2 = (K.X[i,1] - K.Y[j,1])^2 + (K.X[i,2] - K.Y[j,2])^2 + (K.X[i,3] - K.Y[j,3])^2
    d = sqrt(d2)
    return inv(4π*d)
end
function Base.getindex(K::LaplaceMatrix,I::UnitRange,J::UnitRange)
    T = eltype(K)
    m = length(I)
    n = length(J)
    Xv = view(K.X,I,:)
    Yv = view(K.Y,J,:)
    out = Matrix{T}(undef,m,n)
    for j in 1:n
        for i in 1:m
            d2 = (Xv[i,1] - Yv[j,1])^2
            d2 += (Xv[i,2] - Yv[j,2])^2
            d2 += (Xv[i,3] - Yv[j,3])^2
            d = sqrt(d2)
            out[i,j] = inv(4*π*d)
        end
    end
    return out
end
function Base.getindex(K::LaplaceMatrix,i::Int,J::UnitRange)
    T = eltype(K)
    n = length(J)
    Yv = view(K.Y,J,:)
    out = Vector{T}(undef,n)
    for j in 1:n
        d2 =  (K.X[i,1] - Yv[j,1])^2
        d2 += (K.X[i,2] - Yv[j,2])^2
        d2 += (K.X[i,3] - Yv[j,3])^2
        d = sqrt(d2)
        out[j] = inv(4*π*d)
    end
    return out
end
function Base.getindex(K::LaplaceMatrix,I::UnitRange,j::Int)
    T = eltype(K)
    m = length(I)
    Xv = view(K.X,I,:)
    out = Vector{T}(undef,m)
    for i in 1:m
        d2 =  (Xv[i,1] - K.Y[j,1])^2
        d2 += (Xv[i,2] - K.Y[j,2])^2
        d2 += (Xv[i,3] - K.Y[j,3])^2
        d = sqrt(d2)
        out[i] = inv(4*π*d)
    end
    return out
end

"""
    struct LaplaceMatrixVec{T,Td} <: AbstractKernelMatrix{T}

Similar to [`LaplaceMatrix`](@ref), but vectorized.
"""
struct LaplaceMatrixVec{T,Td} <: AbstractKernelMatrix{T}
    X::Matrix{Td}
    Y::Matrix{Td}
    function LaplaceMatrixVec{T}(X::Matrix{Td},Y::Matrix{Td}) where {T,Td}
        @assert size(X,2) == size(Y,2) == 3
        new{T,Td}(X,Y)
    end
end

function LaplaceMatrixVec{T}(_X::Vector{SVector{3,Td}},_Y::Vector{SVector{3,Td}}) where {T,Td}
    X = reshape(reinterpret(Td,_X), 3,:) |> transpose |> collect
    Y = reshape(reinterpret(Td,_Y), 3,:) |> transpose |> collect
    LaplaceMatrixVec{T}(X,Y)
end
LaplaceMatrixVec(args...) = LaplaceMatrixVec{Float64}(args...) # default to Float64

function Base.getindex(K::LaplaceMatrixVec{T},i::Int,j::Int)::T where {T}
    d2 = (K.X[i,1] - K.Y[j,1])^2 + (K.X[i,2] - K.Y[j,2])^2 + (K.X[i,3] - K.Y[j,3])^2
    d = sqrt(d2)
    return inv(4π*d)
end
function Base.getindex(K::LaplaceMatrixVec,I::UnitRange,J::UnitRange)
    T = eltype(K)
    m = length(I)
    n = length(J)
    Xv = view(K.X,I,:)
    Yv = view(K.Y,J,:)
    out = Matrix{T}(undef,m,n)
    @avx for j in 1:n
        for i in 1:m
            d2 = (Xv[i,1] - Yv[j,1])^2
            d2 += (Xv[i,2] - Yv[j,2])^2
            d2 += (Xv[i,3] - Yv[j,3])^2
            d = sqrt(d2)
            out[i,j] = inv(4*π*d)
        end
    end
    return out
end
function Base.getindex(K::LaplaceMatrixVec,i::Int,J::UnitRange)
    T = eltype(K)
    n = length(J)
    Yv = view(K.Y,J,:)
    out = Vector{T}(undef,n)
    @avx for j in 1:n
        d2 =  (K.X[i,1] - Yv[j,1])^2
        d2 += (K.X[i,2] - Yv[j,2])^2
        d2 += (K.X[i,3] - Yv[j,3])^2
        d = sqrt(d2)
        out[j] = inv(4*π*d)
    end
    return out
end
function Base.getindex(K::LaplaceMatrixVec,I::UnitRange,j::Int)
    T = eltype(K)
    m = length(I)
    Xv = view(K.X,I,:)
    out = Vector{T}(undef,m)
    @avx for i in 1:m
        d2 =  (Xv[i,1] - K.Y[j,1])^2
        d2 += (Xv[i,2] - K.Y[j,2])^2
        d2 += (Xv[i,3] - K.Y[j,3])^2
        d = sqrt(d2)
        out[i] = inv(4*π*d)
    end
    return out
end

"""
    struct HelmholtzMatrix{T,Td,Tk} <: AbstractKernelMatrix{T}

Freespace Greens function for Helmholtz's equation in three dimensions. The type
parameter `T` is the return type. The wavenumber parameter `k::Td` is stored in
the struct, and `Td` is the type used to store the points `X` and `Y`.
"""
struct HelmholtzMatrix{T,Td,Tk} <: AbstractKernelMatrix{T}
    X::Vector{SVector{3,Td}}
    Y::Vector{SVector{3,Td}}
    k::Tk
end
HelmholtzMatrix{T}(X::Vector{SVector{3,Td}},Y::Vector{SVector{3,Td}},k::Tk) where {T,Td,Tk} = HelmholtzMatrix{T,Td,Tk}(X,Y,k)
HelmholtzMatrix(args...) = HelmholtzMatrix{ComplexF64}(args...)

function Base.getindex(K::HelmholtzMatrix{T},i::Int,j::Int)::T where {T}
    k = K.k
    d = norm(K.X[i]-K.Y[j])
    inv(4π*d)*exp(im*k*d)
end

struct HelmholtzMatrixVec{T,Td,Tk} <: AbstractKernelMatrix{T}
    X::Matrix{Td}
    Y::Matrix{Td}
    k::Tk
    function HelmholtzMatrixVec{T}(X::Matrix{Td},Y::Matrix{Td},k::Tk) where {T,Td,Tk}
        @assert size(X,2) == size(Y,2) == 3
        new{T,Td,Tk}(X,Y,k)
    end
end
HelmholtzMatrixVec(args...) = HelmholtzMatrixVec{ComplexF64}(args...)

function HelmholtzMatrixVec{T}(_X::Vector{SVector{3,Td}},_Y::Vector{SVector{3,Td}},k) where {T,Td}
    X = reshape(reinterpret(Td,_X), 3,:) |> transpose |> collect
    Y = reshape(reinterpret(Td,_Y), 3,:) |> transpose |> collect
    HelmholtzMatrixVec{T}(X,Y,k)
end

function Base.getindex(K::HelmholtzMatrixVec{T},i::Int,j::Int)::T where {T}
    d2 = (K.X[i,1] - K.Y[j,1])^2 + (K.X[i,2] - K.Y[j,2])^2 + (K.X[i,3] - K.Y[j,3])^2
    d  = sqrt(d2)
    return inv(4π*d)*exp(im*K.k*d)
end
function Base.getindex(K::HelmholtzMatrixVec{Complex{T}},I::UnitRange,J::UnitRange) where {T}
    k = K.k
    m = length(I)
    n = length(J)
    Xv = view(K.X,I,:)
    Yv = view(K.Y,J,:)
    # since LoopVectorization does not (yet) support Complex{T} types, we will
    # reinterpret the output as a Matrix{T}, then use views. This can probably be
    # done better.
    out  = Matrix{Complex{T}}(undef,m,n)
    out_T  = reinterpret(T,out)
    out_r = @views out_T[1:2:end,:]
    out_i = @views out_T[2:2:end,:]
    @avx for j in 1:n
        for i in 1:m
            d2 = (Xv[i,1] - Yv[j,1])^2
            d2 += (Xv[i,2] - Yv[j,2])^2
            d2 += (Xv[i,3] - Yv[j,3])^2
            d  = sqrt(d2)
            s,c = sincos(k*d)
            zr = inv(4π*d)*c
            zi = inv(4π*d)*s
            out_r[i,j] = zr
            out_i[i,j] = zi
        end
    end
    return out
end
function Base.getindex(K::HelmholtzMatrixVec{Complex{T}},i::Int,J::UnitRange) where {T}
    k = K.k
    n = length(J)
    Yv = view(K.Y,J,:)
    out  = Vector{Complex{T}}(undef,n)
    out_T  = reinterpret(T,out)
    out_r = @views out_T[1:2:end]
    out_i = @views out_T[2:2:end]
    @avx for j in 1:n
        d2 = (K.X[i,1] - Yv[j,1])^2
        d2 += (K.X[i,2] - Yv[j,2])^2
        d2 += (K.X[i,3] - Yv[j,3])^2
        d  = sqrt(d2)
        s,c = sincos(k*d)
        zr = inv(4π*d)*c
        zi = inv(4π*d)*s
        out_r[j] = zr
        out_i[j] = zi
    end
    return out
end
function Base.getindex(K::HelmholtzMatrixVec{Complex{T}},I::UnitRange,j::Int) where {T}
    k = K.k
    m = length(I)
    Xv = view(K.X,I,:)
    out  = Vector{Complex{T}}(undef,m)
    out_T  = reinterpret(T,out)
    out_r = @views out_T[1:2:end]
    out_i = @views out_T[2:2:end]
    @avx for i in 1:m
        d2 = (Xv[i,1] -  K.Y[j,1])^2
        d2 += (Xv[i,2] - K.Y[j,2])^2
        d2 += (Xv[i,3] - K.Y[j,3])^2
        d  = sqrt(d2)
        s,c = sincos(k*d)
        zr = inv(4π*d)*c
        zi = inv(4π*d)*s
        out_r[i] = zr
        out_i[i] = zi
    end
    return out
end

"""
    struct ElastostaticMatrix{T,Td,Tp} <: AbstractKernelMatrix{T}

Freespace Greens function for elastostatic equation in three dimensions. The
type parameter `T` is the return type. The wavenumber parameters `μ::Tp,λ::Tp`
are stored in the struct, and `Td` is the type used to store the points `X` and
`Y`.
"""
struct ElastostaticMatrix{T,Td,Tp} <: AbstractKernelMatrix{T}
    X::Vector{SVector{3,Td}}
    Y::Vector{SVector{3,Td}}
    μ::Tp
    λ::Tp
end
ElastostaticMatrix{T}(X::Vector{SVector{3,Td}},Y::Vector{SVector{3,Td}},λ::Tp,μ::Tp) where {T,Td,Tp} = ElastostaticMatrix{T,Td,Tp}(X,Y,λ,μ)
ElastostaticMatrix(args...) = ElastostaticMatrix{SMatrix{3,3,Float64,9}}(args...)

function Base.getindex(K::ElastostaticMatrix{T},i::Int,j::Int)::T where {T}
    μ = K.μ
    λ = K.λ
    ν = λ/(2*(μ+λ))
    x = K.X[i]
    y = K.Y[j]
    r = x - y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    ID = SMatrix{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
    return 1/(16π*μ*(1-ν)*d)*((3-4*ν)*ID + RRT/d^2)
end

struct ElastodynamicMatrix{T,Td,Tp} <: AbstractKernelMatrix{T}
    X::Vector{SVector{3,Td}}
    Y::Vector{SVector{3,Td}}
    μ::Tp
    λ::Tp
    ω::Tp
    ρ::Tp
end
ElastodynamicMatrix{T}(X::Vector{SVector{3,Td}},Y::Vector{SVector{3,Td}},λ::Tp,μ::Tp,ω::Tp,ρ::Tp) where {T,Td,Tp} = ElastodynamicMatrix{T,Td,Tp}(X,Y,λ,μ,ω,ρ)
ElastodynamicMatrix(args...) = ElastodynamicMatrix{SMatrix{3,3,ComplexF64,9}}(args...)

function Base.getindex(K::ElastodynamicMatrix{T},i::Int,j::Int)::T where {T}
    x = K.X[i]
    y = K.Y[j]
    μ = K.μ
    λ = K.λ
    ω = K.ω
    ρ = K.ρ
    c1 = sqrt((λ + 2μ)/ρ)
    c2 = sqrt(μ/ρ)
    r = x .- y
    d = norm(r)
    RRT = r*transpose(r) # r ⊗ rᵗ
    s = -im*ω
    z1 = s*d/c1
    z2 = s*d/c2
    α = 4
    ID    = SMatrix{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
    ψ     = exp(-z2)/d + (1+z2)/(z2^2)*exp(-z2)/d - c2^2/c1^2*(1+z1)/(z1^2)*exp(-z1)/d
    chi   = 3*ψ - 2*exp(-z2)/d - c2^2/c1^2*exp(-z1)/d
    return 1/(α*π*μ)*(ψ*ID - chi*RRT/d^2) + x*transpose(y)
    # return 1/(α*π*μ)*(ψ*ID - chi*RRT/d^2)
end
