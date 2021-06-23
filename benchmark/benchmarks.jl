using BenchmarkTools
using PkgBenchmark
using HMatrices
using StaticArrays
using LinearAlgebra

using HMatrices: PartialACA, ACA, TSVD

# declare global const shared by all _benchfiles
const SUITE = BenchmarkGroup()

include("compressor_bench.jl")
include("assembly_bench.jl")
