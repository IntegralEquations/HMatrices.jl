# Benchmark Report for *HMatrices*

## Job Properties
* Time of benchmark: 11 Jun 2022 - 16:0
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
| `["Compressors", "ACA(0.0, 9223372036854775807, 1.0e-6)"]`        |  55.792 ms (5%) |  16.072 ms | 107.21 MiB (1%) |          66 |
| `["Compressors", "PartialACA(0.0, 9223372036854775807, 1.0e-6)"]` | 232.677 μs (5%) |            | 482.52 KiB (1%) |          48 |
| `["Compressors", "TSVD(0.0, 9223372036854775807, 1.0e-6)"]`       | 340.535 ms (5%) |   8.116 ms |  46.04 MiB (1%) |          17 |
| `["Helmholtz", "assemble cpu"]`                                   |   98.224 s (5%) |    6.290 s |  19.36 GiB (1%) |      713824 |
| `["Helmholtz", "assemble threads"]`                               |   25.600 s (5%) |  63.972 ms |  19.36 GiB (1%) |      736857 |
| `["Helmholtz", "gemv cpu"]`                                       | 971.819 ms (5%) |            |   3.05 MiB (1%) |           6 |
| `["Helmholtz", "gemv threads"]`                                   | 187.363 ms (5%) |            |   9.16 MiB (1%) |          56 |
| `["Helmholtz", "lu"]`                                             |   71.784 s (5%) | 812.047 ms |  57.89 GiB (1%) |     8146475 |
| `["HelmholtzVec", "assemble cpu"]`                                |   68.538 s (5%) |    1.053 s |  17.76 GiB (1%) |     1092683 |
| `["HelmholtzVec", "assemble threads"]`                            |   19.406 s (5%) | 305.914 ms |  17.76 GiB (1%) |     1116979 |
| `["Laplace", "assemble cpu"]`                                     |    2.389 s (5%) | 162.949 ms |   4.52 GiB (1%) |      315818 |
| `["Laplace", "assemble threads"]`                                 | 593.524 ms (5%) |            |   4.53 GiB (1%) |      338852 |
| `["Laplace", "gemv cpu"]`                                         | 282.583 ms (5%) |            |   2.29 MiB (1%) |           6 |
| `["Laplace", "gemv threads"]`                                     |  77.979 ms (5%) |            |   5.34 MiB (1%) |          55 |
| `["Laplace", "lu"]`                                               |   25.312 s (5%) | 359.607 ms |  26.82 GiB (1%) |     7637910 |
| `["LaplaceVec", "assemble cpu"]`                                  |    2.500 s (5%) |            |   4.11 GiB (1%) |      343964 |
| `["LaplaceVec", "assemble threads"]`                              | 353.581 ms (5%) |            |   4.11 GiB (1%) |      366996 |

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
       #1-16  3600 MHz    1834241 s       7846 s      82975 s  1574142594 s          0 s
       
  Memory: 251.4216766357422 GB (214158.078125 MB free)
  Uptime: 9.85167325e6 sec
  Load Avg:  2.8  1.77  1.14
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, skylake-avx512)
```