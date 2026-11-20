using Test
using HMatrices
using HMatrices: ThreadsExecutor, SerialExecutor, spawn!
using LinearAlgebra
using Random
using StaticArrays

"""
    random_program!(ex, objects, nsteps, rng)

Spawn `nsteps` pieces of work, each writing one or two random objects
(`Ref{Int}`s) and reading a few objects written before. As in the drivers, an
object is never written after it was read. A piece of work checks that the
objects it reads hold the value written by their last writer in program order,
then writes its step number into the objects it writes. Random `yield`s inside
the work shuffle the execution, so a missing dependency shows up even on a
single thread. Returns the number of failed checks.
"""
function random_program!(ex, objects, nsteps, rng)
    expected = zeros(Int, length(objects)) # value of each object after the work spawned so far
    wasread = falses(length(objects))
    errors = Threads.Atomic{Int}(0)
    for step in 1:nsteps
        writable = findall(!, wasread)
        written = findall(!iszero, expected)
        writes = unique(rand(rng, writable, rand(rng, 1:2)))
        reads = isempty(written) ? Int[] : setdiff(unique(rand(rng, written, rand(rng, 0:3))), writes)
        want = expected[reads]
        nyields = rand(rng, 0:3)
        spawn!(ex, (objects[i] for i in reads), (objects[i] for i in writes)) do
            for _ in 1:nyields
                yield()
            end
            for (i, v) in zip(reads, want)
                objects[i][] == v || Threads.atomic_add!(errors, 1)
            end
            for i in writes
                objects[i][] = step
            end
        end
        expected[writes] .= step
        wasread[reads] .= true
    end
    return errors
end

# whether a piece of work reading the block `H` runs after the one writing its `leaf`
function block_read_waits_for_leaves(ex, H, leaf)
    done, seen = Ref(false), Ref(false)
    spawn!(() -> (sleep(0.1); done[] = true), ex, (), (leaf,))
    spawn!(() -> (seen[] = done[]), ex, (H,), ())
    wait(ex)
    return seen[]
end

@testset "ThreadsExecutor" begin
    @info "testing the executor with $(Threads.nthreads()) threads"
    @testset "random dependencies" begin
        ex = ThreadsExecutor() # reused after each `wait`
        for seed in 1:20
            objects = [Ref(0) for _ in 1:300]
            errors = random_program!(ex, objects, 200, Xoshiro(seed))
            wait(ex)
            @test errors[] == 0
        end
    end

    @testset "errors" begin
        ex = ThreadsExecutor()
        ran = Threads.Atomic{Int}(0)
        objects = [Ref(0) for _ in 1:4]
        for step in 1:1000
            spawn!(ex, (), (objects[mod1(step, 4)],)) do
                step == 10 && error("failure at step 10")
                Threads.atomic_add!(ran, 1)
            end
        end
        err = try
            wait(ex)
            nothing
        catch e
            e
        end
        # the error itself, as with a serial executor
        @test err isa ErrorException
        @test occursin("failure at step 10", sprint(showerror, err))
        # the other pieces of work ran, and none runs after `wait`
        @test ran[] == 999
        sleep(0.05)
        @test ran[] == 999
        # the error is reported once
        spawn!(() -> nothing, ex, (), (objects[1],))
        @test wait(ex) === nothing
    end

    @testset "accesses" begin
        H = assemble_hmatrix(KernelMatrix((x, y) -> exp(-norm(x - y)), rand(SVector{3, Float64}, 400), rand(SVector{3, Float64}, 400)); threads = false)
        leaf = first(HMatrices.leaves(H))
        ex = ThreadsExecutor(; check_accesses = true)
        # a leaf written after a block containing it was read
        spawn!(() -> nothing, ex, (H,), ())
        @test_throws ErrorException spawn!(() -> nothing, ex, (), (leaf,))
        wait(ex)
        # an object written after it was read
        object = Ref(0)
        spawn!(() -> nothing, ex, (object,), ())
        @test_throws ErrorException spawn!(() -> nothing, ex, (), (object,))
        wait(ex)
        # a piece of work reading and writing an object
        spawn!(() -> nothing, ex, (object,), (object,))
        wait(ex)
        # without the check, the same program runs
        ex = ThreadsExecutor()
        spawn!(() -> nothing, ex, (H,), ())
        spawn!(() -> nothing, ex, (), (leaf,))
        wait(ex)
        # an internal block cannot be written
        @test_throws ArgumentError spawn!(() -> nothing, ex, (), (H,))
        wait(ex)
        # a read of a block waits for the writers of its leaves
        @test block_read_waits_for_leaves(ex, H, leaf)
    end

    @testset "drivers which throw" begin
        # the work spawned before the error is done when the call returns
        ex = ThreadsExecutor()
        done = Threads.Atomic{Int}(0)
        err = try
            HMatrices._run!(ex) do
                for i in 1:100
                    spawn!(() -> (sleep(0.001); Threads.atomic_add!(done, 1)), ex, (), (Ref(i),))
                end
                error("driver failure")
            end
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test done[] == 100
        @test wait(ex) === nothing
    end

    @testset "serial executor" begin
        ex = SerialExecutor()
        out = Int[]
        for i in 1:3
            spawn!(() -> push!(out, i), ex, (), ())
        end
        wait(ex)
        @test out == 1:3
        @test_throws ErrorException spawn!(() -> error("failure"), ex, (), ())
    end
end
