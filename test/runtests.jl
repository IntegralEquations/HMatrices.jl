using SafeTestsets

@safetestset "Hyperrectangle" begin include("hyperrectangle_test.jl") end

@safetestset "Hilbert curve" begin include("hilbertcurve_test.jl") end

@safetestset "Clustertree" begin include("clustertree_test.jl") end

@safetestset "Lowrank matrices" begin include("lowrankmatrices_test.jl") end

@safetestset "Compressors" begin include("compressor_test.jl") end

@safetestset "HMatrix" begin include("hmatrix_test.jl") end
