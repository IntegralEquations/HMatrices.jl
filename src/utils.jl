"""
    debug(flag=true)

Activate debugging messages by setting the environment variable `JULIA_DEBUG` to
`HMatrices`. If `flag=false` deactive debugging messages.
"""
function debug(flag::Bool=true)
    if flag
        ENV["JULIA_DEBUG"] = "HMatrices"
    else
        ENV["JULIA_DEBUG"] = ""
    end
end

"""
    getblocks(filter,hmat,[isterminal=true])

Return all the blocks of `hmat` satisfying `filter(block)::Bool`. If
`isterminal`, do not consider childrens of a block for which
`filter(block)==true`.
"""
function getblocks(f,tree,isterminal=true)
    blocks = Vector{typeof(tree)}()
    getblocks!(f,blocks,tree,isterminal)
end

"""
    getblocks!(filter,blocks,hmat,[isterminal=true])

Like [`getblocks`](@ref), but append valid blocks to `blocks`.
"""
function getblocks!(f,blocks,tree,isterminal)
    if f(tree)
        push!(blocks,tree)
        # terminate the search along this path if terminal=true
        isterminal || map(x->getblocks!(f,blocks,x,isterminal),getchildren(tree))
    else
        # continue on on children
        map(x->getblocks!(f,blocks,x,isterminal),tree.children)
    end
    return blocks
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
        depth(node.parent,acc+1)
    end
end

"""
    @hprofile

A macro which
- resets the default `TimerOutputs.get_defaulttimer` to zero
- execute the code block
- print the profiling details

This is useful as a coarse-grained profiling strategy in `HMatrices`
to get a rough idea of where time is spent. Note that this relies on
`TimerOutputs` annotations manually inserted in the code.
"""
macro hprofile(block)
    return quote
        TimerOutputs.enable_debug_timings(HMatrices)
        reset_timer!()
        $(esc(block))
        print_timer()
    end
end

"""
    @hassert

Assertion which is manually turned off when not in debug mode; i.e. when
`ENV[JULIA_DEBUG]!="HMatrices"`.

# Examples
```julia
HMatrices.debug(false)
@hassert true == false # turned off
HMatrices.debug(true)
@hassert true == false # asssertion error
```
"""
macro hassert(block)
    if haskey(ENV,"JULIA_DEBUG") && ENV["JULIA_DEBUG"]=="HMatrices"
        out = quote
            @assert begin
            $(esc(block))
            end
        end
    else
       :(nothing)
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
