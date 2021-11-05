using Test
using HMatrices
using StaticArrays

A  = rand(12,12)
Tb = SMatrix{2,2,Float64,4}

Ab = BlockArray(A,Tb)

@test HMatrices.blocksize(Ab) == (2,2)
@test size(Ab) == (6,6)
@test eltype(Ab) == Tb

Tb = SMatrix{1,1,Float64,1}

Ab = BlockArray(A,Tb)

@btime sum(view($Ab,1:3,1:3))
@btime sum(view($A,1:3,1:3))

@btime sum($Ab)
@btime sum($A)


Tb = SVector{2,Float64}

Ab = BlockArray(A,Tb)

@test HMatrices.blocksize(Ab) == (2,2)
@test size(Ab) == (6,6)
@test eltype(Ab) == Tb

A  = rand(12,2)
Tb = SMatrix{2,2,Float64,4}

Ab = BlockArray(A,Tb)

@test HMatrices.blocksize(Ab) == (2,2)
@test size(Ab) == (6,6)
@test eltype(Ab) == Tb
