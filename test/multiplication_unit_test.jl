using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

using HMatrices: RkMatrix

HMatrices.debug(false)

Random.seed!(1)

m = 1000
n = 1000
r = 4

T = Float64

R = RkMatrix(rand(m,r),rand(n,r))
F = rand(T,m,n)

X    = rand(SVector{3,Float64},m)
Y    = [rand(SVector{3,Float64}) .+ (1,0,0) for _  in 1:n]
splitter  = CardinalitySplitter(nmax=40)
Xclt      = ClusterTree(X,splitter)
Yclt      = ClusterTree(Y,splitter)
adm       = StrongAdmissibilityStd(eta=3)
rtol      = 1e-5
comp      = PartialACA(rtol=rtol)
K         = HMatrices.LaplaceMatrix(X,Y)
H         = HMatrix(K,Xclt,Yclt,adm,comp;threads=false,distributed=false)

H_full = Matrix(H)
R_full = Matrix(R)

##

################################################################################
##  mul!(C,A,B,α,β)
################################################################################
@testset "mul! unit tests" begin
    α = rand()-0.5
    β = 1.0

    ## 1.1.1: β*full + α*full*full
    @testset "1.1.1" begin
        C  = deepcopy(F)
        tmp = α*F*F + β*C
        HMatrices._mul111!(C,F,F,α)
        @test C ≈ tmp
    end
    ## 1.1.2: β*full + full*sparse
    @testset "1.1.2" begin
        C  = deepcopy(F)
        tmp = β*C + α*F*R_full
        HMatrices._mul112!(C,F,R,α)
        @test C ≈ tmp
    end
    ## 1.1.3: β*full + full*hierarchical
    @testset "1.1.3" begin
        C  = deepcopy(F)
        tmp = β*C + α*F*H_full
        HMatrices._mul113!(C,F,H,α)
        @test C ≈ tmp
    end

    ## 1.1.3b: β*full + adjoint(full)*hierarchical
    @testset "1.1.3b" begin
        C  = deepcopy(F)
        tmp = β*C + α*adjoint(F)*H_full
        HMatrices._mul113!(C,adjoint(F),H,α)
        @test C ≈ tmp
    end
    ## 1.2.1: β*full + sparse*full
    @testset "1.2.1" begin
        C  = deepcopy(F)
        tmp = β*C + α*R_full*F
        HMatrices._mul121!(C,R,F,α)
        @test C ≈ tmp
    end

    ## 1.2.1b: β*full + adjoint(sparse)*full
    @testset "1.2.1b" begin
        C  = deepcopy(F)
        tmp = β*C + α*adjoint(R_full)*F
        HMatrices._mul121!(C,adjoint(R),F,α)
        @test C ≈ tmp
    end
    ## 1.2.2: β*full + sparse*sparse
    @testset "1.2.2" begin
        C  = deepcopy(F)
        tmp = β*C + α*R_full*R_full
        HMatrices._mul122!(C,R,R,α)
        @test C ≈ tmp
    end
    ## 1.2.3: β*full + sparse*hierarchical
    @testset "1.2.3" begin
        C  = deepcopy(F)
        tmp = β*C + α*R_full*H_full
        HMatrices._mul123!(C,R,H,α)
        @test C ≈ tmp
    end

    ## 1.3.1: β*full + hierarchical*full
    @testset "1.3.1" begin
        C  = deepcopy(F)
        tmp = β*C + α*H_full*F
        HMatrices._mul131!(C,H,F,α)
        @test C ≈ tmp
    end

    ## 1.3.2: β*full + hierarchical*sparse
    @testset "1.3.2" begin
        C  = deepcopy(F)
        tmp = β*C + α*H_full*R_full
        HMatrices._mul132!(C,H,R,α)
        @test C ≈ tmp
    end

    ## 1.3.3: β*full + hierarchical*hierarchical
    # should not happen!
    @testset "1.3.3" begin
        C  = deepcopy(F)
        tmp = β*C + α*H_full*H_full
        HMatrices._mul133!(C,H,H,α)
        @test C ≈ tmp
    end
    ## 2.1.1: β*sparse + α*full*full
    @testset "2.1.1" begin
        C  = deepcopy(R);
        C_full = Matrix(C)
        tmp = β*C_full + α*F*F
        HMatrices._mul211!(C,F,F,α)
        @test C ≈ tmp
    end

    ## 2.1.2: β*sparse + α*full*sparse
    @testset "2.1.2" begin
        C  = deepcopy(R);
        C_full = Matrix(C)
        tmp = β*C_full + α*F*R_full
        HMatrices._mul212!(C,F,R,α);
        @test C ≈ tmp
    end

    ## 2.1.3: β*sparse + α*full*hierarchical
    @testset "2.1.3" begin
        C  = deepcopy(R);
        tmp = β*R_full + α*F*H_full;
        HMatrices._mul213!(C,F,H,α)
        @test C ≈ tmp
    end

    ## 2.2.1: β*sparse + α*sparse*dense
    @testset "2.2.1" begin
        C  = deepcopy(R);
        tmp = β*R_full + α*R_full*F
        HMatrices._mul221!(C,R,F,α);
        @test C ≈ tmp
    end

    ## 2.2.2: β*sparse + α*sparse*sparse
    @testset "2.2.2" begin
        C  = deepcopy(R);
        tmp = β*R_full + α*R_full*R_full
        HMatrices._mul222!(C,R,R,α)
        @test C ≈ tmp
    end

    ## 2.2.3: β*sparse + α*sparse*hierarchical
    @testset "2.2.3" begin
        C  = deepcopy(R);
        tmp = β*R_full + α*R_full*H_full
        HMatrices._mul223!(C,R,H,α)
        @test C ≈ tmp
    end

    ## 2.3.1: β*sparse + α*hierarchical*full
    @testset "2.3.1" begin
        C  = deepcopy(R);
        tmp = β*R_full + α*H_full*F
        HMatrices._mul231!(C,H,F,α)
        @test C ≈ tmp
    end

    ## 2.3.2: β*sparse + α*hierarchical*sparse
    @testset "2.3.2" begin
        C  = deepcopy(R);
        tmp = β*R_full + α*H_full*R_full
        HMatrices._mul232!(C,H,R,α)
        @test C ≈ tmp
    end

    ## 2.3.3: β*sparse + α*hierarchical*hierarchical
    @testset "2.2.3" begin
        C  = deepcopy(R);
        tmp = β*R_full + α*H_full*H_full
        HMatrices._mul233!(C,H,H,α)
        @test C ≈ tmp
    end

    ## 3.1.1: β*hierarchical + α*full*full
    @testset "3.1.1" begin
        C  = deepcopy(H);
        tmp = β*H_full + α*F*F
        HMatrices._mul311!(C,F,F,α)
        @test Matrix(C) ≈ tmp
    end

    ## 3.1.2: β*hierarchical + α*full*sparse
    @testset "3.1.2" begin
        C  = deepcopy(H);
        tmp = β*H_full + α*F*R_full
        HMatrices._mul312!(C,F,R,α)
        @test Matrix(C) ≈ tmp
    end

    ## 3.1.3: β*hierarchical + α*full*hierarchical
    @testset "3.1.3" begin
        C  = deepcopy(H);
        tmp = β*H_full + α*F*H_full
        HMatrices._mul313!(C,F,H,α)
        @test Matrix(C) ≈ tmp
    end

    ## 3.2.1: β*hierarchical + α*sparse*full
    @testset "3.2.1" begin
        C  = deepcopy(H);
        tmp = β*H_full + α*R_full*F
        HMatrices._mul321!(C,R,F,α)
        @test Matrix(C) ≈ tmp
    end

    ## 3.2.2: β*hierarchical + α*sparse*sparse
    @testset "3.2.2." begin
        C  = deepcopy(H);
        tmp = β*H_full + α*R_full*R_full
        HMatrices._mul322!(C,R,R,α)
        @test Matrix(C) ≈ tmp
    end

    ## 3.2.3: β*hierarchical + α*sparse*hierarchical

    @testset "3.2.3" begin
        C  = deepcopy(H);
        tmp = β*H_full + α*R_full*H_full
        HMatrices._mul323!(C,R,H,α)
        @test Matrix(C) ≈ tmp
    end

    ## 3.3.1: β*hierarchical + α*hierarchical*full
    @testset "3.3.1" begin
        C  = deepcopy(H);
        tmp = β*H_full + α*H_full*F
        HMatrices._mul331!(C,H,F,α)
        @test Matrix(C) ≈ tmp
    end
    ## 3.3.2: β*hierarchical + α*hierarchical*sparse
    @testset "3.3.2" begin
        C  = deepcopy(H);
        tmp = β*H_full + α*H_full*R_full
        HMatrices._mul332!(C,H,R,α)
        @test Matrix(C) ≈ tmp
    end

    ## 3.3.3: β*hierarchical + α*hierarchical*hierarchical
    @testset "3.3.3" begin
        C  = deepcopy(H);
        tmp = β*H_full + α*H_full*H_full
        HMatrices._mul333!(C,H,H,α)
        @test Matrix(C) ≈ tmp
    end

    ## hmul!
    β = rand() - 0.5
    @testset "3.3.3" begin
        C  = deepcopy(H);
        tmp = β*H_full + α*H_full*H_full
        HMatrices.hmul!(C,H,H,α,β,identity)
        @test Matrix(C) ≈ tmp
    end
end
