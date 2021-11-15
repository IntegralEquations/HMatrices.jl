# Benchmark Report for *HMatrices*

## Job Properties
* Time of benchmark: 15 Nov 2021 - 12:10
* Package commit: e53a64
* Julia commit: 6aaede
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
| `["Compressors", "ACA(0.0, 9223372036854775807, 1.0e-6)"]`        |  76.567 ms (5%) |   7.685 ms | 107.21 MiB (1%) |          67 |
| `["Compressors", "PartialACA(0.0, 9223372036854775807, 1.0e-6)"]` | 446.299 μs (5%) |            | 419.22 KiB (1%) |          45 |
| `["Compressors", "TSVD(0.0, 9223372036854775807, 1.0e-6)"]`       | 648.363 ms (5%) |   4.974 ms |  46.04 MiB (1%) |          16 |
| `["Helmholtz", "assemble cpu"]`                                   |  166.429 s (5%) |    8.595 s |  19.37 GiB (1%) |      727659 |
| `["Helmholtz", "assemble threads"]`                               |   41.592 s (5%) |    1.555 s |  19.37 GiB (1%) |      750704 |
| `["Helmholtz", "gemv cpu"]`                                       |    4.496 s (5%) |            | 261.00 MiB (1%) |     5620826 |
| `["Helmholtz", "gemv threads"]`                                   |    1.378 s (5%) |            |  63.45 MiB (1%) |     1616343 |
| `["Helmholtz", "lu"]`                                             |  192.001 s (5%) |   22.600 s |  58.16 GiB (1%) |    10800690 |
| `["HelmholtzVec", "assemble cpu"]`                                |  110.399 s (5%) |    4.707 s |  17.76 GiB (1%) |     1106518 |
| `["HelmholtzVec", "assemble threads"]`                            |   33.060 s (5%) |    2.216 s |  17.77 GiB (1%) |     1224735 |
| `["Laplace", "assemble cpu"]`                                     |    7.262 s (5%) |    1.135 s |   4.53 GiB (1%) |      320677 |
| `["Laplace", "assemble threads"]`                                 |    2.039 s (5%) | 408.197 ms |   4.53 GiB (1%) |      343721 |
| `["Laplace", "gemv cpu"]`                                         |    2.130 s (5%) |            | 240.72 MiB (1%) |     5184907 |
| `["Laplace", "gemv threads"]`                                     | 210.446 ms (5%) |            |  57.02 MiB (1%) |     1563342 |
| `["Laplace", "lu"]`                                               |   72.762 s (5%) |    8.118 s |  27.03 GiB (1%) |    10119561 |
| `["LaplaceVec", "assemble cpu"]`                                  |    7.162 s (5%) | 857.205 ms |   4.17 GiB (1%) |     1452739 |
| `["LaplaceVec", "assemble threads"]`                              |    1.979 s (5%) | 468.014 ms |   4.11 GiB (1%) |      371866 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["Compressors"]`
- `["Helmholtz"]`
- `["HelmholtzVec"]`
- `["Laplace"]`
- `["LaplaceVec"]`

## Julia versioninfo
```
Julia Version 1.6.1
Commit 6aaedecc44 (2021-04-23 05:59 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
      Ubuntu 20.04.3 LTS
  uname: Linux 5.4.0-89-generic #100-Ubuntu SMP Fri Sep 24 14:50:10 UTC 2021 x86_64 x86_64
  CPU: Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz: 
                 speed         user         nice          sys         idle          irq
       #1-40   800 MHz     116598 s       3332 s      23649 s  102282768 s          0 s
       
  Memory: 31.03945541381836 GB (8270.078125 MB free)
  Uptime: 256097.0 sec
  Load Avg:  3.52  2.51  1.89
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, skylake-avx512)
```