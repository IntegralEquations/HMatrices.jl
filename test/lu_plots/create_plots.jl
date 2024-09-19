using HMatrices
using LinearAlgebra
using StaticArrays
using JLD2

using CairoMakie

const FILE_PATH = joinpath(ARGS[1], "lu_results.jld2")

results = load(FILE_PATH)["results"]
signequal_result = results["1"]

function nthreads_speedup_plot(data)
    signequal_result = data["1"]
    info = Dict{String,Dict{Int,Float64}}()

    for (nthreads, result) in data
        for (label, time) in result
            if !haskey(info, label)
                info[label] = Dict{Int,Float64}()
            end
            speedup = signequal_result[label] / time
            info[label][parse(Int64, nthreads)] = speedup
        end
    end

    f = Figure()
    ax = Axis(
        f[1, 1];
        title = "Speedup of H-LU depending on number of threads",
        xlabel = "Number of threads",
        ylabel = "Speedup",
    )

    for (label, speedup_info) in info
        sorted_info = sort(collect(pairs(speedup_info)))
        nthreads = [x[1] for x in sorted_info]
        speedups = [x[2] for x in sorted_info]
        lines!(ax, nthreads, speedups; label = "$label")
    end

    perfect_speedup = collect(keys(info[minimum(keys(info))]))  # Assuming the smallest size has all thread counts
    lines!(
        ax,
        perfect_speedup,
        perfect_speedup;
        linestyle = :dash,
        linewidth = 2,
        color = :red,
        label = "Perfect speedup",
    )

    ax.yticks = (minimum(perfect_speedup):1:maximum(perfect_speedup))
    ax.xticks = (minimum(perfect_speedup):1:maximum(perfect_speedup))

    f[1, 2] = Legend(f, ax, "Size"; framevisible = false)

    return save(joinpath(ARGS[1], "nthreads_speedup_plot.png"), f)
end

function nthreads_time_plot(data)
    info = Dict{String,Dict{Int,Float64}}()

    for (nthreads, result) in data
        for (label, time) in result
            if !haskey(info, label)
                info[label] = Dict{Int,Float64}()
            end
            info[label][parse(Int64, nthreads)] = time
        end
    end

    f = Figure(; size = (1280, 2000))
    i = 1
    threads_number = collect(keys(info[minimum(keys(info))]))  # Assuming the smallest size has all thread counts
    for (label, times_info) in info
        ax = Axis(
            f[i, 1];
            title = "Time of H-LU depending on number of threads",
            xlabel = "Number of threads",
            ylabel = "Time(s)",
        )
        sorted_info = sort(collect(pairs(times_info)))
        nthreads = [x[1] for x in sorted_info]
        times = [x[2] for x in sorted_info]
        lines!(ax, nthreads, times; label = "$label")
        ax.yticks = (0:max(div(maximum(times), 10), 0.1):(maximum(times)+1))
        ax.xticks = (minimum(threads_number):1:maximum(threads_number))
        f[i, 2] = Legend(f, ax, "Parameters"; framevisible = false)
        i = i + 1
    end
    return save(joinpath(ARGS[1], "nthreads_time_plot.png"), f)
end

nthreads_speedup_plot(results)
nthreads_time_plot(results)