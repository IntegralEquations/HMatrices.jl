# [Internals](@id internals-section)

The internal functions and types of `HMatrices`, which may change without notice:
those of the cluster trees, the compressors and the assembly below, and those of the
[arithmetic](@ref internals-arithmetic-section) (products, factorizations and their
executors) on their own page.

```@autodocs
Modules = [HMatrices]
Pages = ["HMatrices.jl", "utils.jl", "hyperrectangle.jl", "clustertree.jl", "splitter.jl", "kernelmatrix.jl", "rkmatrix.jl", "compressor.jl", "hmatrix.jl"]
Filter = !ispublicapi
```
