n = 32
A = rand(n, n)
H = A + A' + n * I
F = cholesky(H, NoPivot())

chd11 = H[1:(n÷2), 1:(n÷2)]
chd12 = H[1:(n÷2), (n÷2+1):n]
chd21 = H[(n÷2+1):n, 1:(n÷2)]
chd22 = H[(n÷2+1):n, (n÷2+1):n]

F11 = F.factors[1:(n÷2), 1:(n÷2)]
F12 = F.factors[1:(n÷2), (n÷2+1):n]
F21 = F.factors[(n÷2+1):n, 1:(n÷2)]
F22 = F.factors[(n÷2+1):n, (n÷2+1):n]

cholesky!(chd11)
ldiv!(transpose(UpperTriangular(chd11)), chd12)
# rdiv!(chd[2,1], UpperTriangular(chd[1,1]))
mul!(chd22, adjoint(chd12), chd12, -1, 1)
cholesky!(chd22)

@show norm(UpperTriangular(F11 - chd11), Inf)
@show norm(F12 - chd12, Inf)
@show norm(F21 - chd21, Inf)
@show norm(UpperTriangular(F22 - chd22), Inf)
