# Benchmark Report for *HMatrices*

## Job Properties
* Time of benchmark: 11 Jun 2022 - 16:25
* Package commit: 007341
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
| `["Compressors", "ACA(0.0, 9223372036854775807, 1.0e-6)"]`        |  60.896 ms (5%) |  16.845 ms | 114.87 MiB (1%) |          70 |
| `["Compressors", "PartialACA(0.0, 9223372036854775807, 1.0e-6)"]` | 215.466 μs (5%) |            | 451.02 KiB (1%) |          46 |
| `["Compressors", "TSVD(0.0, 9223372036854775807, 1.0e-6)"]`       | 344.458 ms (5%) |   8.569 ms |  46.04 MiB (1%) |          17 |
| `["Helmholtz", "assemble cpu"]`                                   |   96.266 s (5%) |    5.449 s |  19.36 GiB (1%) |      713824 |
| `["Helmholtz", "assemble threads"]`                               |   24.179 s (5%) |  96.356 ms |  19.36 GiB (1%) |      736860 |
| `["Helmholtz", "gemv cpu"]`                                       | 949.268 ms (5%) |            |   3.05 MiB (1%) |           6 |
| `["Helmholtz", "gemv threads"]`                                   | 181.576 ms (5%) |            |   9.16 MiB (1%) |          55 |
| `["Helmholtz", "lu"]`                                             |   66.773 s (5%) | 763.845 ms |  57.89 GiB (1%) |     8146509 |
| `["HelmholtzVec", "assemble cpu"]`                                |   69.864 s (5%) |    1.057 s |  17.76 GiB (1%) |     1092683 |
| `["HelmholtzVec", "assemble threads"]`                            |   20.562 s (5%) | 310.865 ms |  17.76 GiB (1%) |     1116981 |
| `["Laplace", "assemble cpu"]`                                     |    2.493 s (5%) | 161.411 ms |   4.52 GiB (1%) |      315818 |
| `["Laplace", "assemble threads"]`                                 | 604.592 ms (5%) |            |   4.53 GiB (1%) |      338851 |
| `["Laplace", "gemv cpu"]`                                         | 272.634 ms (5%) |            |   2.29 MiB (1%) |           6 |
| `["Laplace", "gemv threads"]`                                     |  76.399 ms (5%) |            |   5.34 MiB (1%) |          55 |
| `["Laplace", "lu"]`                                               |   26.139 s (5%) |    1.065 s |  26.82 GiB (1%) |     7637938 |
| `["LaplaceVec", "assemble cpu"]`                                  |    2.300 s (5%) |            |   4.11 GiB (1%) |      343964 |
| `["LaplaceVec", "assemble threads"]`                              | 740.280 ms (5%) |            |   4.11 GiB (1%) |      366996 |

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
       #1-16  1200 MHz    1843921 s       7865 s      84009 s  1574372975 s          0 s
       
  Memory: 251.4216766357422 GB (207606.51171875 MB free)
  Uptime: 9.85318111e6 sec
  Load Avg:  2.4  1.4  0.98
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, skylake-avx512)
```