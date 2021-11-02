using SafeTestsets

@safetestset "Utils" begin include("utils_test.jl") end

@safetestset "RkMatrix" begin include("rkmatrix_test.jl") end

@safetestset "Compressors" begin include("compressor_test.jl") end

@safetestset "HMatrix" begin include("hmatrix_test.jl") end

@safetestset "Addition" begin include("addition_test.jl") end

@safetestset "Multiplication" begin
    include("multiplication_unit_test.jl")
end
