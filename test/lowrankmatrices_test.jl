using HMatrices
using LinearAlgebra
using Test
using HMatrices: RkMatrix, compression_ratio

@testset "RkMatrix" begin
    m = 20
    n = 30
    r = 10
    A = rand(ComplexF64,m,r)
    B = rand(ComplexF64,n,r)
    R  = RkMatrix(A,B)

    ## basic tests
    @test size(R) == (m,n)
    @test rank(R) == r
    @test R ≈ A*adjoint(B)
    @test compression_ratio(R) ≈ m*n / (r*(m+n))

    # matrix-vector multiplication
    x = rand(ComplexF64,n)
    y = similar(x,m)
    mul!(y,R,x)
    @test y   ≈ R.A*R.Bt*x
    @test R*x ≈ R.A*R.Bt*x

    ## Rkmatrix + Rkmatrix --> Rkmatrix
    # Ap  = rand(m,r)
    # Bpt = rand(r,n)
    # Rp  = RkMatrix(Ap,Bpt)
    # out = Matrix(R) + Matrix(Rp)
    # tmp = R + Rp
    # @test tmp ≈ out
    # @test tmp isa RkMatrix

    ## Rkmatrix + Matrix --> Matrix
    # M   = rand(m,n)
    # out = M+Matrix(R)
    # tmp = M+R
    # @test out ≈ tmp
    # @test tmp isa Matrix
    # @test R + M + Rp ≈ Matrix(R) + M + Matrix(Rp)
    # @test R + Rp + M ≈ Matrix(R) + M + Matrix(Rp)

    ## Rkmatrix*Rkmatrix --> Rkmatrix
    # rp = 5
    # p  = 80
    # Rp  = RkMatrix(rand(n,rp),rand(rp,p))
    # out = Matrix(R)*Matrix(Rp)
    # tmp = R*Rp
    # rank(R)
    # rank(Rp)
    # @test out ≈ tmp
    # tmp isa RkMatrix
    # rank(tmp)
    # rank(tmp) == min(rp,r)

    ## Rkmatrix*Matrix --> Rkmatrix
    # p   = 30
    # M   = rand(p,m)
    # out = M*Matrix(R)
    # tmp = M*R
    # @test out ≈ tmp
    # @test tmp isa RkMatrix
    # M   = rand(n,p)
    # out = Matrix(R)*M
    # tmp = R*M
    # @test out ≈ tmp
    # @test tmp isa RkMatrix

    ## Rkmatrix*vector --> vector
    # x   = rand(n)
    # out = Matrix(R)*x
    # tmp = R*x
    # @test out ≈ tmp
    # @test tmp isa Vector
    # y   = similar(x,m)
    # mul!(y,R,x)
    # @test y ≈ R*x

    ## SVD
    # m = 10; n = 20; r = 5
    # R    = rand(RkMatrix,m,n,r)
    # F    = svd(R)
    # @test F.U*Diagonal(F.S)*F.Vt ≈ R
    # @test F.U' * F.U ≈ Matrix(I,r,r)
    # @test F.Vt * F.Vt' ≈ Matrix(I,r,r)
    # m = 20; n = 10; r = 5
    # R    = rand(RkMatrix,m,n,r)
    # F    = svd(R)
    # @test F.U*Diagonal(F.S)*F.Vt ≈ R
    # @test F.U' * F.U ≈ Matrix(I,r,r)
    # @test F.Vt * F.Vt' ≈ Matrix(I,r,r)
    # m = 20; n = 10; r = 21
    # R    = rand(RkMatrix,m,n,r)
    # F    = svd(R)
    # @test F.U*Diagonal(F.S)*F.Vt ≈ R
    # @test F.U' * F.U ≈ Matrix(I,length(F.S),length(F.S))
    # @test F.Vt * F.Vt' ≈ Matrix(I,length(F.S),length(F.S))

    ## truncation
    # n = 100; m = 200; r = 300
    # U = rand(m,r)
    # S = [exp(-n^2) for n=1:r]
    # Vt = rand(r,n)
    # R    = RkMatrix(U*Diagonal(S),Vt)
    # r0 = 5
    # Rtrunc = HMatrices._trunc_max_rank(R,r0)
    # @test size(Rtrunc) == size(R)
    # @test rank(Rtrunc) == r0
    # @test R ≈ Rtrunc
    # Rtrunc = HMatrices._trunc_atol(R,1e-10)
    # @test Rtrunc ≈ R

    # M = R; atol = 1e-10
    # m,n = size(M)
    # s   = rank(M)
    # QA, RA = LinearAlgebra.qr(M.A)
    # QB, RB = LinearAlgebra.qr(transpose(M.Bt))
    # F      = LinearAlgebra.svd(RA*transpose(RB))
    # r      = findfirst(x -> x<atol, F.S)
    # isnothing(r) && return M # no compression possible
    # if m<n
    #     A  =  QA*(F.U[:,1:r]*LinearAlgebra.Diagonal(F.S[1:r]))
    #     Bt =  F.Vt[1:r,:]*transpose(QB)
    #     Bt = QB*transpose(F.Vt)[:,1:r] |> transpose |> Matrix
    # else
    #     A  =  QA*F.U[:,1:r]
    #     Bt =  (Diagonal(F.S[1:r])*F.Vt[1:r,:])*transpose(QB)
    #     tmp = QB*(transpose(F.Vt)[:,1:r]*Diagonal(F.S[1:r])) |> transpose |> Matrix
    # end

    ## concatenation
    m = 10; n = 5; p = 8; r=3
    R1 = rand(RkMatrix,m,n,r)
    R2 = rand(RkMatrix,m,p,r)
    @test hcat(R1,R2) ≈ hcat(Matrix(R1),Matrix(R2))
    R1 = rand(RkMatrix,m,n,r)
    R2 = rand(RkMatrix,p,n,r)
    @test vcat(R1,R2) ≈ vcat(Matrix(R1),Matrix(R2))
end
