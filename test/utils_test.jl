using Test
using HMatrices
using HMatrices: hilbert_linear_to_cartesian, hilbert_cartesian_to_linear

@testset "Hilbert points" begin
    # check that hilbert curves spans the lattice, and the inverse is correct
    for n in [2^p for p in 0:3]
        lattice = Set((i,j) for i in 0:n-1,j in 0:n-1)
        for d in 0:(n^2-1)
            x,y = hilbert_linear_to_cartesian(n,d)
            pop!(lattice,(x,y))
            @test hilbert_cartesian_to_linear(n,x,y) == d
        end
        @test isempty(lattice)
    end
end
