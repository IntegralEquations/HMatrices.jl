const HLU = LU{<:Any, <:HMatrix}

function Base.getproperty(LU::HLU, s::Symbol)
    H = getfield(LU, :factors) # the underlying hierarchical matrix
    if s == :L
        return UnitLowerTriangular(H)
    elseif s == :U
        return UpperTriangular(H)
    elseif s == :p
        return rowperm(H)
    elseif s == :q
        return colperm(H)
    else
        return getfield(LU, s)
    end
end

Base.propertynames(::HLU, private::Bool = false) = (:L, :U, :p, :q, (private ? fieldnames(LU) : ())...)

function Base.show(io::IO, ::MIME"text/plain", LU::HLU)
    H = getfield(LU, :factors) # the underlying hierarchical matrix
    return println(io, "LU factorization of $H")
end
