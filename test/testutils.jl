using HMatrices
using LoopVectorization
using StaticArrays
using LinearAlgebra
using Gmsh

const EPS = 1e-8

# some simple implementation of laplace and helmholtz single layer matrices.
function laplace_matrix(X, Y)
    f = (x, y) -> begin
        d = norm(x - y) + EPS
        inv(4π * d)
    end
    return KernelMatrix(f, X, Y)
end

function helmholtz_matrix(X, Y, k)
    f = (x, y) -> begin
        EPS = 1e-8 # fudge factor to avoid division by zero
        d = norm(x - y) + EPS
        exp(im * k * d) * inv(4π * d)
    end
    return KernelMatrix(f, X, Y)
end

function elastostatic_matrix(X, Y, μ, λ)
    f = (x, y) -> begin
        ν = λ / (2 * (μ + λ))
        r = x - y
        d = norm(r) + EPS
        RRT = r * transpose(r) # r ⊗ rᵗ
        return 1 / (16π * μ * (1 - ν) * d) * ((3 - 4 * ν) * I + RRT / d^2)
    end
    return KernelMatrix(f, X, Y)
end

function elastosdynamic_matrix(X, Y, μ, λ, ω, ρ)
    f =
        (x, y) -> begin
            c1 = sqrt((λ + 2μ) / ρ)
            c2 = sqrt(μ / ρ)
            r = x - y
            d = norm(r) + EPS
            RRT = r * transpose(r) # r ⊗ rᵗ
            s = -im * ω
            z1 = s * d / c1
            z2 = s * d / c2
            α = 4
            ψ =
                exp(-z2) / d + (1 + z2) / (z2^2) * exp(-z2) / d -
                c2^2 / c1^2 * (1 + z1) / (z1^2) * exp(-z1) / d
            chi = 3 * ψ - 2 * exp(-z2) / d - c2^2 / c1^2 * exp(-z1) / d
            return 1 / (α * π * μ) * (ψ * I - chi * RRT / d^2) + x * transpose(y)
        end
    return KernelMatrix(f, X, Y)
end

"""
    struct LaplaceMatrixVec{T,Td} <: AbstractKernelMatrix{T}

Vectorized version of [`laplace_matrix`](@ref).
"""
struct LaplaceMatrixVec{T,Td} <: AbstractKernelMatrix{T}
    X::Matrix{Td}
    Y::Matrix{Td}
    function LaplaceMatrixVec{T}(X::Matrix{Td}, Y::Matrix{Td}) where {T,Td}
        @assert size(X, 2) == size(Y, 2) == 3
        return new{T,Td}(X, Y)
    end
end

Base.size(K::LaplaceMatrixVec) = size(K.X, 1), size(K.Y, 1)

# constructor based on Vector of StaticVector
function LaplaceMatrixVec{T}(
    _X::Vector{SVector{3,Td}},
    _Y::Vector{SVector{3,Td}},
) where {T,Td}
    X = collect(transpose(reshape(reinterpret(Td, _X), 3, :)))
    Y = collect(transpose(reshape(reinterpret(Td, _Y), 3, :)))
    return LaplaceMatrixVec{T}(X, Y)
end
LaplaceMatrixVec(args...) = LaplaceMatrixVec{Float64}(args...) # default to Float64

function HMatrices.getblock!(out, K::LaplaceMatrixVec, I::UnitRange, J::UnitRange)
    m = length(I)
    n = length(J)
    Xv = view(K.X, I, :)
    Yv = view(K.Y, J, :)
    @turbo for j in 1:n
        for i in 1:m
            d2 = (Xv[i, 1] - Yv[j, 1])^2
            d2 += (Xv[i, 2] - Yv[j, 2])^2
            d2 += (Xv[i, 3] - Yv[j, 3])^2
            d = sqrt(d2) + EPS
            out[i, j] = inv(4 * π * d)
        end
    end
    return out
end
function Base.getindex(K::LaplaceMatrixVec{T}, i::Int, j::Int)::T where {T}
    d2 = (K.X[i, 1] - K.Y[j, 1])^2 + (K.X[i, 2] - K.Y[j, 2])^2 + (K.X[i, 3] - K.Y[j, 3])^2
    d = sqrt(d2) + EPS
    return inv(4π * d)
end

"""
    struct HelmholtzMatrix{T,Td,Tk} <: AbstractKernelMatrix{T}

Vectorized version of [`helmholtz_matrix`](@ref).
"""
struct HelmholtzMatrixVec{T,Td,Tk} <: AbstractKernelMatrix{T}
    X::Matrix{Td}
    Y::Matrix{Td}
    k::Tk
    function HelmholtzMatrixVec{T}(X::Matrix{Td}, Y::Matrix{Td}, k::Tk) where {T,Td,Tk}
        @assert size(X, 2) == size(Y, 2) == 3
        return new{T,Td,Tk}(X, Y, k)
    end
end
HelmholtzMatrixVec(args...) = HelmholtzMatrixVec{ComplexF64}(args...)

Base.size(K::HelmholtzMatrixVec) = size(K.X, 1), size(K.Y, 1)

function HelmholtzMatrixVec{T}(
    _X::Vector{SVector{3,Td}},
    _Y::Vector{SVector{3,Td}},
    k,
) where {T,Td}
    X = collect(transpose(reshape(reinterpret(Td, _X), 3, :)))
    Y = collect(transpose(reshape(reinterpret(Td, _Y), 3, :)))
    return HelmholtzMatrixVec{T}(X, Y, k)
end

function Base.getindex(K::HelmholtzMatrixVec{T}, i::Int, j::Int)::T where {T}
    d2 = (K.X[i, 1] - K.Y[j, 1])^2 + (K.X[i, 2] - K.Y[j, 2])^2 + (K.X[i, 3] - K.Y[j, 3])^2
    d = sqrt(d2) + EPS
    return inv(4π * d) * exp(im * K.k * d)
end
function Base.getindex(
    K::HelmholtzMatrixVec{Complex{T}},
    I::UnitRange,
    J::UnitRange,
) where {T}
    k = K.k
    m = length(I)
    n = length(J)
    Xv = view(K.X, I, :)
    Yv = view(K.Y, J, :)
    # since LoopVectorization does not (yet) support Complex{T} types, we will
    # reinterpret the output as a Matrix{T}, then use views. This can probably be
    # done better.
    out = Matrix{Complex{T}}(undef, m, n)
    out_T = reinterpret(T, out)
    out_r = @views out_T[1:2:end, :]
    out_i = @views out_T[2:2:end, :]
    @avx for j in 1:n
        for i in 1:m
            d2 = (Xv[i, 1] - Yv[j, 1])^2
            d2 += (Xv[i, 2] - Yv[j, 2])^2
            d2 += (Xv[i, 3] - Yv[j, 3])^2
            d = sqrt(d2) + EPS
            s, c = sincos(k * d)
            zr = inv(4π * d) * c
            zi = inv(4π * d) * s
            out_r[i, j] = zr
            out_i[i, j] = zi
        end
    end
    return out
end
function Base.getindex(K::HelmholtzMatrixVec, I::UnitRange, j::Int)
    return vec(K[I, j:j])
end
function Base.getindex(adjK::Adjoint{<:Any,<:HelmholtzMatrixVec}, I::UnitRange, j::Int)
    K = parent(adjK)
    return vec(conj!(K[j:j, I]))
end

function points_on_sphere(npts, R = 1)
    theta = π * rand(npts)
    phi = 2 * π * rand(npts)
    x = @. sin(theta) * cos(phi)
    y = @. R * sin(theta) * sin(phi)
    z = @. R * cos(theta)
    data = vcat(x', y', z')
    pts = collect(reinterpret(SVector{3,Float64}, vec(data)))
    return pts
end

function points_on_cylinder(n, radius, shift = SVector(0, 0, 0))
    step = 1.75 * π * radius / sqrt(n)
    result = Vector{SVector{3,Float64}}(undef, n)
    length = 2 * π * radius
    pointsPerCircle = length / step
    angleStep = 2 * π / pointsPerCircle
    for i in 0:(n-1)
        x = radius * cos(angleStep * i)
        y = radius * sin(angleStep * i)
        z = step * i / pointsPerCircle
        result[i+1] = shift + SVector(x, y, z)
    end
    return result
end

"""
    points_on_airplain(meshsize)

Returns a vector of 3D points of A319 airplane. 
mshesize=92.2 -> points_size=100501
mshesize=345 -> points_size=10318
"""
function points_on_airplain(meshsize)
    gmsh.initialize()
    path_to_file = joinpath(HMatrices.PROJECT_ROOT, "test", "airplane", "A319.geo")
    gmsh.open(path_to_file)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    gmsh.model.mesh.generate(2)
    node_tags, coords, pcoords = gmsh.model.mesh.getNodes()
    points = [SVector{3,Float64}(coords[i:i+2]) for i in 1:3:length(coords)]
    gmsh.finalize()
    return points
end
