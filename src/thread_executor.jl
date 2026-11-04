"""
    SerialExecutor()

Executor running every piece of work immediately, in the calling task: with
`executor = SerialExecutor()`, the algorithms which take an `executor` (e.g.
[`lu!`](@ref), [`cholesky!`](@ref)) run sequentially, as with `threads = false`.
"""
struct SerialExecutor end

spawn!(work, ::SerialExecutor, reads, writes) = (work(); nothing)
Base.wait(::SerialExecutor) = nothing

"""
    mutable struct ExecutorTask

A piece of work of a [`ThreadsExecutor`](@ref), with the number of its unfinished
dependencies (one more while it is being spawned, so that it does not run before),
and the pieces of work depending on it. `work === nothing` for a join.
"""
mutable struct ExecutorTask
    work::Any
    @atomic number_of_unfinished_dependencies::Int
    const lock::Threads.SpinLock # protects `is_done` and `successors`
    is_done::Bool
    successors::Vector{ExecutorTask}
end
ExecutorTask(work) = ExecutorTask(work, 1, Threads.SpinLock(), false, ExecutorTask[])

"""
    ThreadsExecutor(; check_accesses = false)

Executor running pieces of work as tasks on Julia's threads, each one waiting only for
the last piece of work which wrote each of the objects it reads or writes: with
`executor = ThreadsExecutor()`, the algorithms which take an `executor` (e.g.
[`lu!`](@ref), [`cholesky!`](@ref)) run in parallel, as with `threads = true`. An error
raised by a piece of work is rethrown as is, without the backtrace of its task; running
sequentially (see [`SerialExecutor`](@ref)) gives the full backtrace.

`spawn!(work, ex, reads, writes)` registers `work()`, which reads the objects of
`reads` and modifies those of `writes`, and `wait(ex)` returns once all the work
spawned is done, then forgets it, and rethrows the first error of a piece of work.

There is no general dependency analysis: the executor only keeps the last writer of
every object, which is enough for drivers satisfying the following access pattern:
- an object is written by a sequence of pieces of work, which are run in the order
  in which they are spawned;
- an object is only read once final, i.e. after its last write has been spawned, so
  that there are no write-after-read hazards.
The objects written are leaves of an `HMatrix`, or other objects (e.g. a low-rank
term computed by one piece of work and read by others). A read of an internal block
of an `HMatrix` is a read of all its leaves: it waits for a join over the last
writers of its leaves, created at its first read and reused afterwards. With
`check_accesses = true`, the executor checks this access pattern: that no object is
written after it, or an internal block containing it, was read.

A piece of work never blocks: it counts its unfinished dependencies, and is spawned
as a task by the piece of work completing the last of them. The objects are
identified by their `objectid`, so that the executor does not keep them alive.
"""
mutable struct ThreadsExecutor
    const last_writers::Dict{UInt, Union{Nothing, ExecutorTask}} # by objectid
    const read_objects::Base.IdSet{Any} # kept only if `check_accesses`
    const check_accesses::Bool
    const number_of_unfinished_tasks::Threads.Atomic{Int}
    const all_done::Threads.Condition
    @atomic exception::Any # the first error of a piece of work
end
function ThreadsExecutor(; check_accesses = false)
    return ThreadsExecutor(
        Dict{UInt, Union{Nothing, ExecutorTask}}(), Base.IdSet{Any}(), check_accesses, Threads.Atomic{Int}(0),
        Threads.Condition(), nothing,
    )
end

"""
    _stored(x)

The object which an access to `x` reads or writes: the `HMatrix` underlying an
adjoint, a `Hermitian` or a triangle of one, and `x` itself otherwise.
"""
_stored(x) = x
_stored(x::Union{Adjoint, Hermitian, HTriangular}) = _stored(parent(x))

_isinternal(x) = x isa HMatrix && !isleaf(x)

function spawn!(work, ex::ThreadsExecutor, reads, writes)
    for x in writes
        x = _stored(x)
        _isinternal(x) && throw(ArgumentError("a piece of work writes an internal block of an HMatrix"))
        ex.check_accesses && _check_write(ex, x)
    end
    task = _new_task!(ex, work)
    for x in reads
        x = _stored(x)
        ex.check_accesses && push!(ex.read_objects, x)
        _add_dependency!(task, _last_writer!(ex, x))
    end
    for x in writes
        x = _stored(x)
        _add_dependency!(task, _last_writer!(ex, x))
        ex.last_writers[objectid(x)] = task
    end
    _release!(ex, task)
    return nothing
end

function Base.wait(ex::ThreadsExecutor)
    @lock ex.all_done while ex.number_of_unfinished_tasks[] > 0
        wait(ex.all_done)
    end
    empty!(ex.last_writers)
    empty!(ex.read_objects)
    exception = @atomicswap ex.exception = nothing
    isnothing(exception) || throw(exception)
    return nothing
end

function _new_task!(ex::ThreadsExecutor, work)
    Threads.atomic_add!(ex.number_of_unfinished_tasks, 1)
    return ExecutorTask(work)
end

"""
    _last_writer!(ex::ThreadsExecutor, x)

The piece of work after which `x` is up to date, if any. A read of an internal block
of an `HMatrix`, which is final, waits for a join over the last writers of its
leaves, created once.
"""
function _last_writer!(ex::ThreadsExecutor, x)
    key = objectid(x)
    haskey(ex.last_writers, key) && return ex.last_writers[key]
    _isinternal(x) || return nothing
    writers = [ex.last_writers[objectid(leaf)] for leaf in leaves(x) if haskey(ex.last_writers, objectid(leaf))]
    join = nothing
    if !isempty(writers)
        join = _new_task!(ex, nothing)
        foreach(w -> _add_dependency!(join, w), writers)
        _release!(ex, join)
    end
    return ex.last_writers[key] = join
end

"""
    _check_write(ex::ThreadsExecutor, x)

Throw an error if `x`, or an internal block of an `HMatrix` containing it, was already
read: its readers would not wait for this write.
"""
function _check_write(ex::ThreadsExecutor, x)
    node = x
    while true
        node in ex.read_objects && error("an object is written after it, or a block containing it, was read")
        node isa HMatrix && !isroot(node) || return nothing
        node = parentnode(node)
    end
    return
end

"""
    _add_dependency!(task::ExecutorTask, dependency)

Make `task` wait for `dependency` (an `ExecutorTask`, or `nothing`), unless it is
done.
"""
function _add_dependency!(task::ExecutorTask, dependency)
    (isnothing(dependency) || dependency === task) && return task
    @lock dependency.lock if !dependency.is_done
        @atomic task.number_of_unfinished_dependencies += 1
        push!(dependency.successors, task)
    end
    return task
end

"""
    _release!(ex::ThreadsExecutor, task::ExecutorTask)

Decrement the number of unfinished dependencies of `task`, and run it once it
reaches zero: a join is finished at once, the other pieces of work are spawned as
tasks.
"""
function _release!(ex::ThreadsExecutor, task::ExecutorTask)
    (@atomic task.number_of_unfinished_dependencies -= 1) == 0 || return nothing
    isnothing(task.work) ? _finish!(ex, task) : Threads.@spawn(_execute!($ex, $task))
    return nothing
end

"""
    _execute!(ex::ThreadsExecutor, task::ExecutorTask)

Run the work of `task`, recording its error if it throws, then release the pieces of
work depending on it.
"""
function _execute!(ex::ThreadsExecutor, task::ExecutorTask)
    try
        task.work()
    catch e
        @atomicreplace ex.exception nothing => e
    end
    # a finished piece of work must not keep alive what its work captured (e.g. low-rank terms)
    task.work = nothing
    return _finish!(ex, task)
end

function _finish!(ex::ThreadsExecutor, task::ExecutorTask)
    successors = @lock task.lock begin
        task.is_done = true
        s, task.successors = task.successors, ExecutorTask[]
        s
    end
    foreach(s -> _release!(ex, s), successors)
    # under the lock, so that `wait` cannot miss the notification between its check and its wait
    Threads.atomic_sub!(ex.number_of_unfinished_tasks, 1) == 1 && @lock ex.all_done notify(ex.all_done)
    return nothing
end

"""
    _executor(threads::Bool)

The default executor: a [`ThreadsExecutor`](@ref) if `threads` and Julia runs on
several threads, and a [`SerialExecutor`](@ref) otherwise.
"""
_executor(threads::Bool) = threads && Threads.nthreads() > 1 ? ThreadsExecutor() : SerialExecutor()

"""
    _run!(driver, executor)

Run `driver()`, which spawns work on `executor`, then wait for this work, also when
the driver throws, so that no work outlives the call.
"""
function _run!(driver, executor)
    result = try
        driver()
    catch
        try
            wait(executor)
        catch
        end
        rethrow()
    end
    wait(executor)
    return result
end

"""
    workspaces(T)

An empty pool of workspaces of type `T`, for [`with_workspace`](@ref).
"""
workspaces(T) = Base.Lockable(T[], Threads.SpinLock())

"""
    with_workspace(f, pool, make)

`f(w)` with a workspace `w` used by no other piece of work at the same time: taken from
`pool` (see [`workspaces`](@ref)), or made by `make()` if the pool is empty, then put
back in `pool`. A piece of work never waits for a workspace, and the pool grows to the
number of pieces of work running at the same time.
"""
function with_workspace(f, pool, make)
    w = @lock pool (isempty(pool[]) ? nothing : pop!(pool[]))
    w = isnothing(w) ? make() : w
    result = f(w)
    @lock pool push!(pool[], w)
    return result
end
