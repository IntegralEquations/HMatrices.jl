"""
    struct HyperRectangle{N,T}

Axis-aligned hyperrectangle in `N` dimensions given by `low_corner::SVector{N,T}` and
`high_corner::SVector{N,T}`.
"""
struct HyperRectangle{N,T}
    low_corner::SVector{N,T}
    high_corner::SVector{N,T}
end
HyperRectangle(l::Tuple, h::Tuple) = HyperRectangle(SVector(l), SVector(h))
HyperRectangle(l::SVector, h::SVector) = HyperRectangle(promote(l, h)...)
# 1d case
HyperRectangle(a::Number, b::Number) = HyperRectangle(SVector(a), SVector(b))

# Base.eltype(::HyperRectangle{N,T}) where {N,T} = T
# ambient_dimension(::HyperRectangle{N}) where {N} = N
# geometric_dimension(::HyperRectangle{N}) where {N} = N
low_corner(r::HyperRectangle) = r.low_corner
high_corner(r::HyperRectangle) = r.high_corner

function Base.isapprox(h1::HyperRectangle, h2::HyperRectangle; kwargs...)
    return isapprox(h1.low_corner, h2.low_corner; kwargs...) &&
           isapprox(h1.high_corner, h2.high_corner; kwargs...)
end

width(r::HyperRectangle) = high_corner(r) - low_corner(r)

half_width(r::HyperRectangle) = width(r) / 2

"""
    distance(Ω1,Ω2)

Minimal Euclidean distance between a point `x ∈ Ω1` and `y ∈ Ω2`.
"""
function distance(rec1::HyperRectangle{N}, rec2::HyperRectangle{N}) where {N}
    d2 = 0
    rec1_low_corner = low_corner(rec1)
    rec1_high_corner = high_corner(rec1)
    rec2_low_corner = low_corner(rec2)
    rec2_high_corner = high_corner(rec2)
    for i in 1:N
        d2 +=
            max(0, rec1_low_corner[i] - rec2_high_corner[i])^2 +
            max(0, rec2_low_corner[i] - rec1_high_corner[i])^2
    end
    return sqrt(d2)
end

"""
    distance(x::SVector,r::HyperRectangle)

The (minimal) Euclidean distance between the point `x` and any point `y ∈ r`.
"""
function distance(pt::SVector{N}, rec::HyperRectangle{N}) where {N}
    d2 = 0
    rec_low_corner = low_corner(rec)
    rec_high_corner = high_corner(rec)
    for i in 1:N
        d2 += max(0, pt[i] - rec_high_corner[i])^2 + max(0, rec_low_corner[i] - pt[i])^2
    end
    return sqrt(d2)
end
distance(rec::HyperRectangle{N}, pt::SVector{N}) where {N} = distance(pt, rec)

"""
    diameter(Ω)

Largest distance between `x` and `y` for `x,y ∈ Ω`.
"""
diameter(r::HyperRectangle) = norm(high_corner(r) .- low_corner(r), 2)

"""
    radius(Ω)

Half the [`diameter`](@ref).
"""
radius(r::HyperRectangle) = diameter(r) / 2

"""
    center(Ω)

Center of the smallest ball containing `Ω`.
"""
center(r::HyperRectangle) = (low_corner(r) + high_corner(r)) / 2

Base.in(point, h::HyperRectangle) = all(low_corner(h) .<= point .<= high_corner(h))

function vertices(rec::HyperRectangle{2})
    lc = low_corner(rec)
    hc = high_corner(rec)
    return SVector(
        SVector(lc[1], lc[2]),
        SVector(hc[1], lc[2]),
        SVector(hc[1], hc[2]),
        SVector(lc[1], hc[2]),
    )
end
function vertices(rec::HyperRectangle{3})
    lc = low_corner(rec)
    hc = high_corner(rec)
    return SVector(
        # lower face
        SVector(lc[1], lc[2], lc[3]),
        SVector(hc[1], lc[2], lc[3]),
        SVector(hc[1], hc[2], lc[3]),
        SVector(lc[1], hc[2], lc[3]),
        # upper face
        SVector(lc[1], lc[2], hc[3]),
        SVector(hc[1], lc[2], hc[3]),
        SVector(hc[1], hc[2], hc[3]),
        SVector(lc[1], hc[2], hc[3]),
    )
end

function bounding_box(els, cube = false)
    isempty(els) && (error("data cannot be empty"))
    lb = center(first(els))
    ub = center(first(els))
    for el in els
        pt = center(el)
        lb = min.(lb, pt)
        ub = max.(ub, pt)
    end
    if cube # fit a square/cube instead
        w = maximum(ub - lb)
        xc = (ub + lb) / 2
        # min/max below helps avoid floating point issues with results with the
        # cube not being exactly contained in the original rectangle
        lb = min.(xc .- w / 2, lb)
        ub = max.(xc .+ w / 2, ub)
        # TODO: return HyperCube instead
    end
    lb == ub && (lb = prevfloat.(lb); ub = nextfloat.(ub)) # to avoid "empty" rectangles
    return HyperRectangle(lb, ub)
end
center(x::SVector) = x
center(x::NTuple) = SVector(x)

"""
    split(rec::HyperRectangle,[axis]::Int,[place])

Split a hyperrectangle in two along the `axis` direction at the  position
`place`. Returns a tuple with the two resulting hyperrectangles.

When no `place` is given, defaults to splitting in the middle of the axis.

When no axis and no place is given, defaults to splitting along the largest axis.
"""
function Base.split(rec::HyperRectangle{N}, axis, place) where {N}
    rec_low_corner = low_corner(rec)
    rec_high_corner = high_corner(rec)
    high_corner1 = SVector(ntuple(n -> n == axis ? place : rec_high_corner[n], N))
    low_corner2 = SVector(ntuple(n -> n == axis ? place : rec_low_corner[n], N))
    rec1 = HyperRectangle(rec_low_corner, high_corner1)
    rec2 = HyperRectangle(low_corner2, rec_high_corner)
    return (rec1, rec2)
end
function Base.split(rec::HyperRectangle, axis)
    place = (high_corner(rec)[axis] + low_corner(rec)[axis]) / 2
    return split(rec, axis, place)
end
function Base.split(rec::HyperRectangle)
    axis = argmax(high_corner(rec) .- low_corner(rec))
    return split(rec, axis)
end
