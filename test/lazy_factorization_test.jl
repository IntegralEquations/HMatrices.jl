using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays

using HMatrices: RkMatrix, PartialACA, RandomizedSVD, ClusterTree, CardinalitySplitter, StrongAdmissibilityStd, flush!, flush_tree!, children, ThreadsExecutor, SerialExecutor, LazyDriver, LazyLowRankTerm, CarryTerm

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

Random.seed!(1)

m = 1500
X = rand(SVector{3, Float64}, m)
splitter = CardinalitySplitter(; nmax = 50)
Xclt = ClusterTree(X, splitter)
adm = StrongAdmissibilityStd(; eta = 3)
K = laplace_matrix(X, X)
H = assemble_hmatrix(K, Xclt, Xclt; adm, comp = PartialACA(; atol = 1.0e-8), threads = false)
H_full = Matrix(H)

relerr(A, B) = norm(A - B) / norm(B)
same(A, B) = relerr(A, B) <= 1.0e-12

@testset "Lazy updates" begin
    for comp in (PartialACA(; atol = 1.0e-6, recompress = false), PartialACA(; atol = 1.0e-6), RandomizedSVD(; atol = 1.0e-6))
        @testset "$comp" begin
            @testset "single product" begin
                C = deepcopy(H)
                d = LazyDriver(C, H, H, comp)
                push!(d, C, H, H)
                flush_tree!(d, C)
                @test relerr(Matrix(C), H_full - H_full * H_full) < 1.0e-5
                @test HMatrices.isclean(C)
                @test isempty(d.pending)
            end

            @testset "several products and a carry" begin
                C = deepcopy(H)
                d = LazyDriver(C, H, H, comp)
                A = RkMatrix(rand(m, 3), rand(m, 3))
                push!(HMatrices.pending!(d, C).carry, CarryTerm(LazyLowRankTerm(copy(A.A), copy(A.B)), 0, 0))
                push!(d, C, H, H)
                push!(d, C, H, H)
                flush_tree!(d, C)
                exact = H_full + Matrix(A) - 2 * H_full * H_full
                @test relerr(Matrix(C), exact) < 1.0e-5
            end

            @testset "updates pushed at different times and levels" begin
                C = deepcopy(H)
                d = LazyDriver(C, H, H, comp)
                push!(d, C, H, H)
                flush!(d, C) # only the root: work is handed to the children
                @test !haskey(d.pending, C)
                @test any(chd -> haskey(d.pending, chd), children(C))
                # register a second product directly on the children
                chdC, chdH = children(C), children(H)
                for i in 1:2, j in 1:2, k in 1:2
                    push!(d, chdC[i, j], chdH[i, k], chdH[k, j])
                end
                flush_tree!(d, C)
                @test isempty(d.pending)
                @test relerr(Matrix(C), H_full - 2 * H_full * H_full) < 1.0e-5
            end
        end
    end
end


@testset "exact low-rank products" begin
    # complex data, to check the conjugations with a complex coefficient
    Kc = single_layer_sphere(1500; k = 4)
    cltc = ClusterTree(Kc.X, CardinalitySplitter(; nmax = 50))
    Hc = assemble_hmatrix(Kc, cltc, cltc; adm = StrongAdmissibilityStd(3), comp = PartialACA(; atol = 1.0e-8), threads = false)
    isrk(X) = HMatrices.isleaf(X) && HMatrices.isadmissible(X)
    α = 0.3 + 0.7im
    found = Dict{Symbol, Int}()
    for X in HMatrices.nodes(Hc), Y in HMatrices.nodes(Hc)
        (HMatrices.isleaf(X) || HMatrices.isleaf(Y)) || continue
        size(X, 2) == size(Y, 1) && HMatrices.colrange(X) == HMatrices.rowrange(Y) || continue
        HMatrices._has_low_rank_form(X, Y) || continue
        kind = isrk(X) && isrk(Y) ? :rkrk : isrk(X) ? :rkh : isrk(Y) ? :hrk : :dd
        get(found, kind, 0) >= 3 && continue
        found[kind] = get(found, kind, 0) + 1
        A, B = HMatrices._compute_low_rank_term(X, Y, α)
        @test size(A, 2) == HMatrices._low_rank_term_rank(X, Y)
        @test A * B' ≈ α * HMatrices._tree_matrix(X) * HMatrices._tree_matrix(Y)
        # the same product with adjoint factors
        A, B = HMatrices._compute_low_rank_term(adjoint(Y), adjoint(X), α)
        @test A * B' ≈ α * (HMatrices._tree_matrix(X) * HMatrices._tree_matrix(Y))'
        if kind == :rkrk
            @test size(A, 2) == min(rank(HMatrices.data(X)), rank(HMatrices.data(Y)))
        end
        all(k -> get(found, k, 0) >= 3, (:rkrk, :rkh, :hrk, :dd)) && break
    end
    @test all(k -> get(found, k, 0) >= 1, (:rkrk, :rkh, :hrk, :dd))
end

"""
    ShuffleExecutor(seed)

Test executor which records the work spawned by a driver with its dependencies
(from the declared accesses, as `ThreadsExecutor` does), and runs it in `wait`, on
the calling task, in a random topological order. A driver which does not declare
an access correctly gives results depending on the order.
"""
struct ShuffleExecutor
    seed::Int
    lastwriter::Dict{UInt, Int}
    work::Vector{Any}
    preds::Vector{Vector{Int}}
end
ShuffleExecutor(seed) = ShuffleExecutor(seed, Dict{UInt, Int}(), Any[], Vector{Int}[])

function _writers(ex::ShuffleExecutor, x)
    haskey(ex.lastwriter, objectid(x)) && return [ex.lastwriter[objectid(x)]]
    x isa HMatrix && !HMatrices.isleaf(x) || return Int[]
    return [ex.lastwriter[objectid(l)] for l in HMatrices.leaves(x) if haskey(ex.lastwriter, objectid(l))]
end

function HMatrices.spawn!(f, ex::ShuffleExecutor, reads, writes)
    id = length(ex.work) + 1
    accessed = map(HMatrices._stored, collect(Iterators.flatten((reads, writes))))
    push!(ex.preds, unique(reduce(vcat, (_writers(ex, x) for x in accessed); init = Int[])))
    foreach(x -> ex.lastwriter[objectid(HMatrices._stored(x))] = id, writes)
    push!(ex.work, f)
    return nothing
end

function Base.wait(ex::ShuffleExecutor)
    n = length(ex.work)
    succs = [Int[] for _ in 1:n]
    npending = length.(ex.preds)
    for (i, p) in enumerate(ex.preds), j in p
        push!(succs[j], i)
    end
    ready = findall(iszero, npending)
    rng = Xoshiro(ex.seed)
    while !isempty(ready)
        k = rand(rng, eachindex(ready))
        i = ready[k]
        ready[k] = ready[end]
        pop!(ready)
        ex.work[i]()
        for j in succs[i]
            (npending[j] -= 1) == 0 && push!(ready, j)
        end
    end
    @assert all(iszero, npending)
    empty!(ex.work), empty!(ex.preds), empty!(ex.lastwriter)
    return nothing
end

@testset "products and solves on threads" begin
    # the drivers give the same results serially and on threads, where the executor checks
    # their accesses
    comp = PartialACA(; atol = 1.0e-8)
    Hh = assemble_hmatrix(Hermitian(K), Xclt, Xclt; adm, comp, threads = false)
    F = lu(H, comp)
    L, U = UnitLowerTriangular(F.factors), UpperTriangular(F.factors)
    serial_and_threads(f) = same(Matrix(f(ThreadsExecutor(; check_accesses = true))), Matrix(f(SerialExecutor())))
    @test serial_and_threads(ex -> HMatrices._hmul_tree!(deepcopy(H), H, H, -1, 1, comp; executor = ex))
    @test serial_and_threads(ex -> HMatrices._hmul_tree!(deepcopy(Hh), H, adjoint(H), -1, 1, comp; executor = ex))
    @test serial_and_threads(ex -> HMatrices.hldiv!(L, deepcopy(H), comp; executor = ex))
    @test serial_and_threads(ex -> HMatrices.hrdiv!(deepcopy(H), U, comp; executor = ex))
    # a `Hermitian` storing its lower triangle
    Cl = HMatrices._hmul_tree!(Hermitian(deepcopy(H), :L), H, adjoint(H), -1, 1, comp)
    Mt = HMatrices._tree_matrix(H)
    @test relerr(HMatrices._tree_matrix(Cl), Hermitian(Mt, :L) - Mt * Mt') < 1.0e-6
    # an error in a piece of work is thrown as such, whatever the executor
    Hn = deepcopy(Hh)
    rmul!(parent(Hn), -1)
    for ex in (SerialExecutor(), ThreadsExecutor())
        @test_throws PosDefException cholesky(Hn, comp; executor = ex)
    end
end

@testset "Factorizations" begin
    # (kernel, wavenumber, size) for single-layer operators on the sphere
    for (name, k, n) in (("Laplace", 0, 4000), ("Helmholtz", 8, 3000))
        @testset "LU, $name" begin
            K = single_layer_sphere(n; k)
            clt = ClusterTree(K.X, CardinalitySplitter(; nmax = 50))
            H = assemble_hmatrix(
                K, clt, clt; adm = StrongAdmissibilityStd(3), comp = PartialACA(; rtol = 1.0e-10),
                threads = false,
            )
            M = Matrix(H)
            y = rand(eltype(M), n)
            exact = M \ y
            atol = 1.0e-8
            for comp in (PartialACA(; atol, recompress = false), PartialACA(; atol), RandomizedSVD(; atol))
                F = lu(H, comp)
                @test HMatrices.isclean(F.factors)
                approx = F \ y
                @test norm(approx - exact) < 1.0e-5 * norm(exact)
                # `F.L` and `F.U` act in the ordering of the cluster tree
                @test F.L * (F.U * approx[F.q]) ≈ y[F.p]
            end
            # the same operations in a different order give the same factors, up to the
            # rounding errors (some BLAS round differently with the memory layout)
            for comp in (PartialACA(; atol), RandomizedSVD(; atol))
                Ms = HMatrices._tree_matrix(lu(H, comp; executor = SerialExecutor()).factors)
                Ft = lu(H, comp; executor = ThreadsExecutor(; check_accesses = true))
                @test HMatrices.isclean(Ft.factors)
                @test same(HMatrices._tree_matrix(Ft.factors), Ms)
                for seed in 1:2
                    @test same(HMatrices._tree_matrix(lu(H, comp; executor = ShuffleExecutor(seed)).factors), Ms)
                end
            end
        end
    end
    # the upper triangle of the stored factor of a Cholesky factorization (the lower
    # leaves of a `Hermitian` have no data)
    function upper_factor(F)
        H = parent(F.factors)
        U = zeros(eltype(H), size(H))
        for leaf in HMatrices.leaves(H)
            HMatrices.hasdata(leaf) || continue
            U[HMatrices.rowrange(leaf), HMatrices.colrange(leaf)] = Matrix(HMatrices.data(leaf))
        end
        return triu(U)
    end
    @testset "Cholesky" begin
        K = single_layer_sphere(4000)
        clt = ClusterTree(K.X, CardinalitySplitter(; nmax = 50))
        H = assemble_hmatrix(Hermitian(K), clt, clt; adm = StrongAdmissibilityStd(3), comp = PartialACA(; rtol = 1.0e-10), threads = false)
        M = Matrix(K)
        y = rand(4000)
        exact = M \ y
        for comp in (PartialACA(; atol = 1.0e-8), RandomizedSVD(; atol = 1.0e-8))
            F = cholesky(H, comp; executor = SerialExecutor())
            @test norm(F \ y - exact) < 1.0e-5 * norm(exact)
            Us = upper_factor(F)
            Ft = cholesky(H, comp; executor = ThreadsExecutor(; check_accesses = true))
            @test same(upper_factor(Ft), Us)
            for seed in 1:2
                Fr = cholesky(H, comp; executor = ShuffleExecutor(seed))
                @test same(upper_factor(Fr), Us)
            end
        end
    end
end
