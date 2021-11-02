# HMatrices.jl

*A package for assembling and doing linear algebra with hierarchical matrices with a focus on boundary integral equations* 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://WaveProp.github.io/HMatrices.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://WaveProp.github.io/HMatrices.jl/dev)
[![Build Status](https://github.com/WaveProp/HMatrices.jl/workflows/CI/badge.svg)](https://github.com/WaveProp/HMatrices.jl/actions)
[![Coverage](https://codecov.io/gh/WaveProp/HMatrices.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/WaveProp/HMatrices.jl)
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-blue.svg)

## Installation
Install from the Pkg REPL:
```
pkg> add https://github.com/WaveProp/HMatrices.jl
```

## Overview

Hierarchical matrices, or ℋ-matrices for short, describe abstract matrices with
a hierarchical (tree-like) block structure. They are commonly used to compress
certain linear operators containing large blocks of low numerical rank. See
[1,2] in the [references](#references) section below for background information
on hierarchical matrices.

This package provides an implementation of various algorithms for assembling and
performing linear algebra with ℋ-matrices with a *focus on applications arising
in boundary integral equation methods*. Thus, many of the examples included in
the documentation will deal with compression of boundary integral operators.

In general, to construct an ℋ-matrix you need:

- a matrix-like object `K` with a `getindex(K,i,j)` method.
- a tree partition of the rows and columns indices of `K`
- an admissibility condition for blocks given by a node of the row tree and a
  node of the column tree

The admissibility condition and the tree partition depend on the application.

## Basic usage

The following simple example illustrates how you may use this package to
compress Laplace's single layer operator on a circle:


```julia
    using HMatrices, Clusters, Plots, LinearAlgebra
    # create some random points
    N           = 10_000 
    pts         = [(cos(θ),sin(θ)) for θ in range(0,stop=2π,length=N)]
    # make a cluster tree
    clustertree = ClusterTree(pts)
    # then a block cluster tree
    blocktree   = BlockTree(clustertree,clustertree)
    # create your matrix
    f(x,y)      =  x==y ? 0.0 : -1/(2π)*log(norm(x-y)) # Laplace single layer kernels in 2d
    M           = [f(x,y) for x in clustertree.data, y in clustertree.data]
    # compress it
    H           = HMatrix(M,blocktree)
```
Many of the steps above accept keyword arguments or functors for modifying their default behavior.

Often one cannot assemble the full matrix. In this case the `LazyMatrix` type is useful:
```julia
    L = LazyMatrix(f,clustertree.data,clustertree.data)
    H = HMatrix(L,blocktree)
```
This is just like the matrix we build `M`, but it computes the entries *on demand* and does not store them. To see the data sparse structure of the hierarchical matrix the package includes a `Plots` recipe so that you can do `plot(H)` to see something like the following image:

![HMatrix](docs/src/figures/hmatrix.png "HMatrix")

## References

[1] Hackbusch, Wolfgang. Hierarchical matrices: algorithms and analysis. Vol. 49. Heidelberg: Springer, 2015.

[2] Bebendorf, Mario. Hierarchical matrices. Springer Berlin Heidelberg, 2008.

