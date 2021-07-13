using SafeTestsets

@safetestset "Hilbert curve" begin include("hilbertcurve_test.jl") end

@safetestset "Lowrank matrices" begin include("lowrankmatrices_test.jl") end

@safetestset "Compressors" begin include("compressor_test.jl") end

@safetestset "HMatrix" begin include("hmatrix_test.jl") end
