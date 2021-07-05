# Hilbert space-filling curve, adapted from C code at https://en.wikipedia.org/wiki/Hilbert_curve

"""
    hilbert_cartesian_to_linear(n,x,y)

Convert the cartesian indices `x,y` into a linear index `d` using a hilbert
curve of order `n`. The coordinates `x,y` range from `0` to `n-1`, and the
output `d` ranges from `0` to `n^2-1`.

See [https://en.wikipedia.org/wiki/Hilbert_curve](https://en.wikipedia.org/wiki/Hilbert_curve).
"""
function hilbert_cartesian_to_linear(n::Integer,x,y)
    @assert ispow2(n)
    @assert 0 ≤ x ≤ n-1
    @assert 0 ≤ y ≤ n-1
    d = 0
    s = n >> 1
    while s>0
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s^2*((3*rx)⊻ry)
        x,y = _rot(n,x,y,rx,ry)
        s = s >> 1
    end
    @assert 0 ≤ d ≤ n^2-1
    return d
end

"""
    hilbert_linear_to_cartesian(n,d)

Convert the linear index `0 ≤ d ≤ n^2-1` into the cartesian coordinates `0 ≤ x <
n-1` and `0 ≤ y ≤ n-1` on the Hilbert curve of order `n`.

See [https://en.wikipedia.org/wiki/Hilbert_curve](https://en.wikipedia.org/wiki/Hilbert_curve).
"""
function hilbert_linear_to_cartesian(n::Integer,d)
    @assert ispow2(n)
    @assert 0 ≤ d ≤ n^2-1
    x,y = 0,0
    s = 1
    while s<n
        rx = 1 & (d >> 1)
        ry = 1 & (d ⊻ rx)
        x,y = _rot(s,x,y,rx,ry)
        x +=  s*rx
        y +=  s*ry
        d = d >> 2
        s = s << 1
    end
    @assert 0 ≤ x ≤ n-1
    @assert 0 ≤ y ≤ n-1
    return x,y
end

# auxiliary function using in hilbert curve. Rotates the points x,y
function _rot(n,x,y,rx,ry)
    if ry == 0
        if rx == 1
            x = n-1 - x
            y = n-1 - y
        end
        x,y = y,x
    end
    return x,y
end

function hilbert_points(n::Integer)
    @assert ispow2(n)
    xx = Int[]
    yy = Int[]
    for d in 0:n^2-1
        x,y = hilbert_linear_to_cartesian(n,d)
        push!(xx,x)
        push!(yy,y)
    end
    xx,yy
end
