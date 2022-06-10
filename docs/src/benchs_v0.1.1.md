# Benchmark Report for *HMatrices*

## Job Properties
* Time of benchmark: 10 Jun 2022 - 17:0
* Package commit: 26fea0
* Julia commit: bf5349
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
| `["Compressors", "ACA(0.0, 9223372036854775807, 1.0e-6)"]`        |  39.932 ms (5%) |            | 107.21 MiB (1%) |          65 |
| `["Compressors", "PartialACA(0.0, 9223372036854775807, 1.0e-6)"]` | 244.804 μs (5%) |            | 482.34 KiB (1%) |          47 |
| `["Compressors", "TSVD(0.0, 9223372036854775807, 1.0e-6)"]`       | 326.827 ms (5%) |            |  46.04 MiB (1%) |          16 |
| `["Helmholtz", "assemble cpu"]`                                   |   88.972 s (5%) | 453.427 ms |  19.36 GiB (1%) |      709578 |
| `["Helmholtz", "assemble threads"]`                               |   24.718 s (5%) | 382.291 ms |  19.36 GiB (1%) |      732613 |
| `["Helmholtz", "gemv cpu"]`                                       | 344.368 ms (5%) |            |  61.60 MiB (1%) |     1616025 |
| `["Helmholtz", "gemv threads"]`                                   | 331.605 ms (5%) |            |  61.60 MiB (1%) |     1616025 |
| `["Helmholtz", "lu"]`                                             |   72.511 s (5%) |    1.862 s |  57.98 GiB (1%) |     8222356 |
| `["HelmholtzVec", "assemble cpu"]`                                |   65.144 s (5%) | 255.618 ms |  17.75 GiB (1%) |     1088437 |
| `["HelmholtzVec", "assemble threads"]`                            |   18.579 s (5%) | 131.085 ms |  17.76 GiB (1%) |     1111471 |
| `["Laplace", "assemble cpu"]`                                     |    2.368 s (5%) |            |   4.52 GiB (1%) |      311572 |
| `["Laplace", "assemble threads"]`                                 | 603.344 ms (5%) |            |   4.53 GiB (1%) |      334605 |
| `["Laplace", "gemv cpu"]`                                         | 124.529 ms (5%) |            |  55.14 MiB (1%) |     1563318 |
| `["Laplace", "gemv threads"]`                                     | 124.629 ms (5%) |            |  55.14 MiB (1%) |     1563318 |
| `["Laplace", "lu"]`                                               |   26.284 s (5%) | 251.976 ms |  26.86 GiB (1%) |     7724374 |
| `["LaplaceVec", "assemble cpu"]`                                  |    1.587 s (5%) |  40.043 ms |   4.11 GiB (1%) |      339717 |
| `["LaplaceVec", "assemble threads"]`                              | 369.461 ms (5%) |            |   4.11 GiB (1%) |      362750 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["Compressors"]`
- `["Helmholtz"]`
- `["HelmholtzVec"]`
- `["Laplace"]`
- `["LaplaceVec"]`

## Julia versioninfo
```
Julia Version 1.7.2
Commit bf53498635 (2022-02-06 15:21 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
      Ubuntu 20.04.4 LTS
  uname: Linux 5.13.0-28-generic #31~20.04.1-Ubuntu SMP Wed Jan 19 14:08:10 UTC 2022 x86_64 x86_64
  CPU: Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz: 
                 speed         user         nice          sys         idle          irq
       #1-16  1200 MHz    1735734 s       7735 s      76700 s  1560999326 s          0 s
       
  Memory: 251.4216766357422 GB (212679.40625 MB free)
  Uptime: 9.7688475e6 sec
  Load Avg:  3.04  2.3  1.65
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, skylake-avx512)
```