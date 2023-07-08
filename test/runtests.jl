using SafeTestsets

@safetestset "Utils" begin
    include("utils_test.jl")
end

@safetestset "HyperRectangle" begin
    include("hyperrectangle_test.jl")
end

@safetestset "ClusterTree" begin
    include("clustertree_test.jl")
end

@safetestset "RkMatrix" begin
    include("rkmatrix_test.jl")
end

@safetestset "Compressors" begin
    include("compressor_test.jl")
end

@safetestset "HMatrix" begin
    include("hmatrix_test.jl")
end

@safetestset "Addition" begin
    include("addition_test.jl")
end

@safetestset "Multiplication" begin
    include("multiplication_test.jl")
end

@safetestset "Triangular" begin
    include("triangular_test.jl")
end

@safetestset "LU" begin
    include("lu_test.jl")
end

# @safetestset "DHMatrix" begin include("dhmatrix_test.jl") end
