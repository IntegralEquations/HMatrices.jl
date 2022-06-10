# Benchmark Report for *HMatrices*

## Job Properties
* Time of benchmark: 10 Jun 2022 - 17:15
* Package commit: dirty
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
| `["Compressors", "ACA(0.0, 9223372036854775807, 1.0e-6)"]`        |  34.053 ms (5%) |            |  99.55 MiB (1%) |          61 |
| `["Compressors", "PartialACA(0.0, 9223372036854775807, 1.0e-6)"]` | 233.181 μs (5%) |            | 482.34 KiB (1%) |          47 |
| `["Compressors", "TSVD(0.0, 9223372036854775807, 1.0e-6)"]`       | 321.884 ms (5%) |            |  46.04 MiB (1%) |          16 |
| `["Helmholtz", "assemble cpu"]`                                   |   89.956 s (5%) | 399.859 ms |  19.36 GiB (1%) |      709578 |
| `["Helmholtz", "assemble threads"]`                               |   24.903 s (5%) | 510.673 ms |  19.36 GiB (1%) |      732613 |
| `["Helmholtz", "gemv cpu"]`                                       |    2.585 s (5%) |            | 390.72 MiB (1%) |     8996780 |
| `["Helmholtz", "gemv threads"]`                                   | 338.144 ms (5%) |            |  61.60 MiB (1%) |     1616025 |
| `["Helmholtz", "lu"]`                                             |   71.828 s (5%) |    1.258 s |  57.99 GiB (1%) |     8572231 |
| `["HelmholtzVec", "assemble cpu"]`                                |   66.737 s (5%) | 372.154 ms |  17.75 GiB (1%) |     1088437 |
| `["HelmholtzVec", "assemble threads"]`                            |   19.034 s (5%) |  38.952 ms |  17.76 GiB (1%) |     1235264 |
| `["Laplace", "assemble cpu"]`                                     |    3.286 s (5%) |  42.410 ms |   4.52 GiB (1%) |      311572 |
| `["Laplace", "assemble threads"]`                                 | 605.793 ms (5%) |            |   4.53 GiB (1%) |      334605 |
| `["Laplace", "gemv cpu"]`                                         | 134.986 ms (5%) |            |  55.14 MiB (1%) |     1563319 |
| `["Laplace", "gemv threads"]`                                     | 125.187 ms (5%) |            |  55.14 MiB (1%) |     1563318 |
| `["Laplace", "lu"]`                                               |   26.886 s (5%) | 336.713 ms |  26.87 GiB (1%) |     8059276 |
| `["LaplaceVec", "assemble cpu"]`                                  |    3.139 s (5%) | 380.169 ms |   4.16 GiB (1%) |     1271792 |
| `["LaplaceVec", "assemble threads"]`                              | 354.316 ms (5%) |            |   4.11 GiB (1%) |      362750 |

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
       #1-16  1200 MHz    1745670 s       7735 s      77177 s  1561132442 s          0 s
       
  Memory: 251.4216766357422 GB (214379.88671875 MB free)
  Uptime: 9.76974509e6 sec
  Load Avg:  3.16  1.99  1.5
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, skylake-avx512)
```