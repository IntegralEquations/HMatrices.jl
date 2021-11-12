# Benchmark Report for *HMatrices*

## Job Properties
* Time of benchmark: 12 Nov 2021 - 18:33
* Package commit: dirty
* Julia commit: ae8452
* Julia command flags: `-O3`
* Environment variables: `JULIA_NUM_THREADS => 4` `OPEN_BLAS_NUM_THREADS => 1`

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                                                | time            | GC time    | memory          | allocations |
|-------------------------------------------------------------------|----------------:|-----------:|----------------:|------------:|
| `["Compressors", "ACA(0.0, 9223372036854775807, 1.0e-6)"]`        |  61.241 ms (5%) |   4.972 ms | 107.21 MiB (1%) |          67 |
| `["Compressors", "PartialACA(0.0, 9223372036854775807, 1.0e-6)"]` | 219.119 μs (5%) |            | 450.72 KiB (1%) |          47 |
| `["Compressors", "TSVD(0.0, 9223372036854775807, 1.0e-6)"]`       | 286.398 ms (5%) |   1.540 ms |  46.04 MiB (1%) |          16 |
| `["Helmholtz", "assemble cpu"]`                                   |    8.748 s (5%) | 274.875 ms |   2.00 GiB (1%) |      150511 |
| `["Helmholtz", "assemble procs"]`                                 |    6.581 s (5%) |            |  20.08 MiB (1%) |      348308 |
| `["Helmholtz", "assemble threads"]`                               |    2.018 s (5%) |            |   2.00 GiB (1%) |      155071 |
| `["Helmholtz", "gemv cpu"]`                                       |  79.233 ms (5%) |            |  13.97 MiB (1%) |      375961 |
| `["Helmholtz", "gemv threads"]`                                   |  54.442 ms (5%) |            |  13.96 MiB (1%) |      375658 |
| `["Laplace", "assemble cpu"]`                                     | 667.362 ms (5%) |            | 661.53 MiB (1%) |       81762 |
| `["Laplace", "assemble procs"]`                                   | 227.796 ms (5%) |            | 778.27 KiB (1%) |       26833 |
| `["Laplace", "assemble threads"]`                                 | 166.750 ms (5%) |            | 661.88 MiB (1%) |       86321 |
| `["Laplace", "gemv cpu"]`                                         |    7.608 s (5%) | 267.259 ms | 893.60 MiB (1%) |    26145811 |
| `["Laplace", "gemv threads"]`                                     |  27.370 ms (5%) |            |  12.56 MiB (1%) |      363514 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["Compressors"]`
- `["Helmholtz"]`
- `["Laplace"]`

## Julia versioninfo
```
Julia Version 1.6.3
Commit ae8452a9e0 (2021-09-23 17:34 UTC)
Platform Info:
  OS: macOS (x86_64-apple-darwin19.5.0)
  uname: Darwin 21.1.0 Darwin Kernel Version 21.1.0: Wed Oct 13 17:33:23 PDT 2021; root:xnu-8019.41.5~1/RELEASE_X86_64 x86_64 i386
  CPU: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz: 
                 speed         user         nice          sys         idle          irq
       #1-16  2300 MHz     746382 s          0 s     346592 s   14103601 s          0 s
       
  Memory: 16.0 GB (3794.78515625 MB free)
  Uptime: 209471.0 sec
  Load Avg:  7.66259765625  5.48193359375  4.224609375
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, skylake)
```