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
        return out
    else
       return :(nothing)
    end
end
