"""
    debug(flag=true)

Activate debugging messages by setting the environment variable `JULIA_DEBUG` to
`HierarchicalMatrices`. If `flag=false` deactive debugging messages.
"""
function debug(flag=true)
    if flag
        @eval ENV["JULIA_DEBUG"] = "HierarchicalMatrices"
    else
        @eval ENV["JULIA_DEBUG"] = ""
    end
end

"""
    depth(node,acc=0)

Recursive function to compute the depth of `node` in a a tree-like structure.
Require the method `getparent(node)` to be implemented. Overload this function
if your structure has a more efficient way to compute `depth` (e.g. if it stores
it in a field).
"""
function depth(node,acc=0)
    if isroot(node)
        return acc
    else
        depth(getparent(node),acc+1)
    end
end

"""
    @hprofile

A macro which
- resets the default `TimerOutputs.get_defaulttimer` to zero
- execute the code block
- print the profiling details

This is useful as a coarse-grained profiling strategy in `HierarchicalMatrices`
to get a rough idea of where time is spent. Note that this relies on
`TimerOutputs` annotations manually inserted in the code.
"""
macro hprofile(block)
    return quote
        reset_timer!()
        $(esc(block))
        print_timer()
    end
end

"""
    svector(f,n)

Like `ntuple` from `Base`, but convert result to `SVector`.
"""
function svector(f,n)
    ntuple(f,n) |> SVector
end

"""
    const Maybe{T}

Type alias to Union{Tuple{},T}.
"""
const Maybe{T}  = Union{Tuple{},T}

"""
    abstractmethod

A method of an `abstract type` for which concrete subtypes are expected
to provide an implementation.
"""
function abstractmethod(T)
    error("this method needs to be implemented by the concrete subtype $(typeof(T)).")
end
