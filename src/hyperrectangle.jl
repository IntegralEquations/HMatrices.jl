"""
    struct HyperRectangle{N,T}

Axis-aligned hyperrectangle in `N` dimensions given by `low_corner::SVector{N,T}` and
`high_corner::SVector{N,T}`.
"""
struct HyperRectangle{N,T}
    low_corner::SVector{N,T}
    high_corner::SVector{N,T}
end
HyperRectangle(lc::SVector,hc::SVector) = HyperRectangle(promote(lc,hc)...)

low_corner(r::HyperRectangle)  = r.low_corner
high_corner(r::HyperRectangle) = r.high_corner
low_corner(r::HyperRectangle,i::Int)  = r.low_corner[i]
high_corner(r::HyperRectangle,i::Int) = r.high_corner[i]

"""
    dimension(r::HyperRectangle)

Return the ambient dimension of the hyperrectangle.
"""
dimension(r::HyperRectangle{N}) where {N} = N

Base.eltype(h::HyperRectangle{N,T}) where {N,T}    = T

function Base.:(==)(h1::HyperRectangle, h2::HyperRectangle)
    (low_corner(h1)  == low_corner(h2)) &&
    (high_corner(h1) == high_corner(h2))
end

function Base.in(x,h::HyperRectangle)
    N = dimension(h)
    xl = low_corner(h)
    xh = high_corner(h)
    all(i -> xl[i] <= x[i] <= xh[i],1:N)
end

"""
    split(rec::HyperRectangle,[axis]::Int,[place])

Split a hyperrectangle in two along the `axis` direction at the  position
`place`. Returns a tuple with the two resulting hyperrectangles.

When no `place` is given, defaults to splitting in the middle of the axis.

When no axis and no place is given, defaults to splitting along the largest axis.
"""
function Base.split(rec::HyperRectangle,axis,place)
    N            = dimension(rec)
    high_corner1 = svector(n-> n==axis ? place : rec.high_corner[n], N)
    low_corner2  = svector(n-> n==axis ? place : rec.low_corner[n], N)
    rec1         = HyperRectangle(rec.low_corner, high_corner1)
    rec2         = HyperRectangle(low_corner2,rec.high_corner)
    return (rec1, rec2)
end
function Base.split(rec::HyperRectangle,axis)
    place        = (rec.high_corner[axis] + rec.low_corner[axis])/2
    split(rec,axis,place)
end
function Base.split(rec::HyperRectangle)
    axis = argmax(rec.high_corner .- rec.low_corner)
    split(rec,axis)
end

"""
    diameter(r::HyperRectangle)

The Euclidean distance between `low_corner(r)` and `high_corner(r)`.
"""
diameter(cub::HyperRectangle) = norm(cub.high_corner - cub.low_corner,2)

"""
    distance(r1::HyperRectangle,r2)

The (minimal) Euclidean distance between a point `x ∈ r1` and `y ∈ r2`.
"""
function distance(rec1::HyperRectangle{N},rec2::HyperRectangle{N}) where {N}
    d2 = 0
    for i=1:N
        d2 += max(0,rec1.low_corner[i] - rec2.high_corner[i])^2 +
              max(0,rec2.low_corner[i] - rec1.high_corner[i])^2
    end
    return sqrt(d2)
end



"""
    HyperRectangle(pts::Vector{<:SVector},cube=false)

Contruct the smallest [`HyperRectangle`](@ref) containing all `pts`. If
`cube=true`, construct instead the smallest hypercube containing all `pts`. Note
that hypercube is not a type in itself, and therefore whether or not a
`HyperRectangle` is a hypercube has to be determined dynamically.
"""
function HyperRectangle{N,T}(pts::AbstractVector{SVector{N,T}},cube::Bool=false) where {N,T}
    lb = reduce((x, v) -> min.(x, v), pts) # lower bound
    ub = reduce((x, v) -> max.(x, v), pts) # upper bound
    if cube # fit a square/cube instead
        w  = maximum(ub-lb)
        xc = (ub + lb) / 2
        lb = xc .- w/2
        ub = xc .+ w/2
    end
    return HyperRectangle(lb,ub)
end
function HyperRectangle(pts::AbstractVector{SVector{N,T}},args...) where {N,T}
    HyperRectangle{N,T}(pts,args...)
end

center(rec::HyperRectangle) = (rec.low_corner + rec.high_corner) ./ 2

"""
    radius(r::HyperRectangle)

Half the [`diameter`](@ref).
"""
radius(rec::HyperRectangle) = diameter(rec) ./ 2
