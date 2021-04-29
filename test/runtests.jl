using SafeTestsets

@safetestset "Hyperrectangle" begin include("hyperrectangle_test.jl") end

@safetestset "Clustertree" begin include("clustertree_test.jl") end

@safetestset "Blocktree" begin include("clustertree_test.jl") end
