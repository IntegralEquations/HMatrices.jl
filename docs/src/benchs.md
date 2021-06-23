# Benchmark Report for *HMatrices*

## Job Properties
* Time of benchmark: 2 May 2021 - 10:7
* Package commit: dirty
* Julia commit: 6aaede
* Julia command flags: `-O3`
* Environment variables: `JULIA_NUM_THREADS => 1` `OPEN_BLAS_NUM_THREADS => 1`

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                                                | time            | GC time    | memory          | allocations |
|-------------------------------------------------------------------|----------------:|-----------:|----------------:|------------:|
| `["Assembly", "Laplace kernel 50000"]`                            |    1.476 s (5%) | 192.301 ms |   1.46 GiB (1%) |      245868 |
| `["Compressors", "ACA(0.0, 9223372036854775807, 1.0e-6)"]`        | 173.728 ms (5%) |  11.996 ms | 206.40 MiB (1%) |          92 |
| `["Compressors", "PartialACA(0.0, 9223372036854775807, 1.0e-6)"]` | 283.944 μs (5%) |            | 419.19 KiB (1%) |          44 |
| `["Compressors", "TSVD(0.0, 9223372036854775807, 1.0e-6)"]`       | 141.331 ms (5%) | 854.505 μs |  46.04 MiB (1%) |          15 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["Assembly"]`
- `["Compressors"]`

## Julia versioninfo
```
Julia Version 1.6.1
Commit 6aaedecc44 (2021-04-23 05:59 UTC)
Platform Info:
  OS: macOS (x86_64-apple-darwin18.7.0)
  uname: Darwin 20.3.0 Darwin Kernel Version 20.3.0: Thu Jan 21 00:07:06 PST 2021; root:xnu-7195.81.3~1/RELEASE_X86_64 x86_64 i386
  CPU: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz: 
                 speed         user         nice          sys         idle          irq
       #1-16  2300 MHz    1073909 s          0 s     570700 s   25879754 s          0 s
       
  Memory: 16.0 GB (2096.8046875 MB free)
  Uptime: 405641.0 sec
  Load Avg:  2.9375  2.85498046875  2.7421875
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, skylake)
```