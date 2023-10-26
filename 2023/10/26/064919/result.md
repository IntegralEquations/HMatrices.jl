# Benchmark result

* Pull request commit: [`db572561a03e08d5d0c224197fedada208aa0602`](https://github.com/WaveProp/HMatrices.jl/commit/db572561a03e08d5d0c224197fedada208aa0602)
* Pull request: <https://github.com/WaveProp/HMatrices.jl/pull/38> (Automate benchmarks)

# Judge result
# Benchmark Report for */home/runner/work/HMatrices.jl/HMatrices.jl*

## Job Properties
* Time of benchmarks:
    - Target: 26 Oct 2023 - 06:48
    - Baseline: 26 Oct 2023 - 06:48
* Package commits:
    - Target: 0132e0
    - Baseline: 0132e0
* Julia commits:
    - Target: bed2cd
    - Baseline: bed2cd
* Julia command flags:
    - Target: None
    - Baseline: None
* Environment variables:
    - Target: None
    - Baseline: None

## Results
A ratio greater than `1.0` denotes a possible regression (marked with :x:), while a ratio less
than `1.0` denotes a possible improvement (marked with :white_check_mark:). Only significant results - results
that indicate possible regressions or improvements - are shown below (thus, an empty table means that all
benchmark results remained invariant between builds).

| ID                                             | time ratio                   | memory ratio |
|------------------------------------------------|------------------------------|--------------|
| `["trigonometry", "circular", ("cos", "π")]`   |                1.12 (5%) :x: |   1.00 (1%)  |
| `["trigonometry", "circular", ("cos", 0.0)]`   | 0.94 (5%) :white_check_mark: |   1.00 (1%)  |
| `["trigonometry", "circular", ("sin", "π")]`   |                1.12 (5%) :x: |   1.00 (1%)  |
| `["trigonometry", "circular", ("tan", "π")]`   |                1.12 (5%) :x: |   1.00 (1%)  |
| `["trigonometry", "hyperbolic", ("cos", 0.0)]` | 0.91 (5%) :white_check_mark: |   1.00 (1%)  |
| `["trigonometry", "hyperbolic", ("sin", "π")]` |                1.06 (5%) :x: |   1.00 (1%)  |
| `["trigonometry", "hyperbolic", ("tan", "π")]` |                1.06 (5%) :x: |   1.00 (1%)  |
| `["trigonometry", "hyperbolic", ("tan", 0.0)]` |                1.15 (5%) :x: |   1.00 (1%)  |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["trigonometry", "circular"]`
- `["trigonometry", "hyperbolic"]`
- `["utf8"]`

## Julia versioninfo

### Target
```
Julia Version 1.9.3
Commit bed2cd540a1 (2023-08-24 14:43 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 22.04.3 LTS
  uname: Linux 6.2.0-1015-azure #15~22.04.1-Ubuntu SMP Fri Oct  6 13:20:44 UTC 2023 x86_64 x86_64
  CPU: Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz: 
              speed         user         nice          sys         idle          irq
       #1  2294 MHz        798 s          0 s        147 s       2039 s          0 s
       #2  2294 MHz       1367 s          0 s        223 s       1381 s          0 s
  Memory: 6.759746551513672 GB (5434.89453125 MB free)
  Uptime: 307.41 sec
  Load Avg:  1.34  0.88  0.4
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, broadwell)
  Threads: 1 on 2 virtual cores
```

### Baseline
```
Julia Version 1.9.3
Commit bed2cd540a1 (2023-08-24 14:43 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 22.04.3 LTS
  uname: Linux 6.2.0-1015-azure #15~22.04.1-Ubuntu SMP Fri Oct  6 13:20:44 UTC 2023 x86_64 x86_64
  CPU: Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz: 
              speed         user         nice          sys         idle          irq
       #1  2294 MHz       1083 s          0 s        176 s       2080 s          0 s
       #2  2294 MHz       1413 s          0 s        231 s       1680 s          0 s
  Memory: 6.759746551513672 GB (5377.80859375 MB free)
  Uptime: 342.93 sec
  Load Avg:  1.25  0.91  0.43
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, broadwell)
  Threads: 1 on 2 virtual cores
```

---
# Target result
# Benchmark Report for */home/runner/work/HMatrices.jl/HMatrices.jl*

## Job Properties
* Time of benchmark: 26 Oct 2023 - 6:48
* Package commit: 0132e0
* Julia commit: bed2cd
* Julia command flags: None
* Environment variables: None

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                             | time            | GC time  | memory          | allocations |
|------------------------------------------------|----------------:|---------:|----------------:|------------:|
| `["trigonometry", "circular", ("cos", "π")]`   |   1.800 ns (5%) |          |                 |             |
| `["trigonometry", "circular", ("cos", 0.0)]`   |   6.000 ns (5%) |          |                 |             |
| `["trigonometry", "circular", ("sin", "π")]`   |   1.900 ns (5%) |          |                 |             |
| `["trigonometry", "circular", ("sin", 0.0)]`   |   6.200 ns (5%) |          |                 |             |
| `["trigonometry", "circular", ("tan", "π")]`   |   1.800 ns (5%) |          |                 |             |
| `["trigonometry", "circular", ("tan", 0.0)]`   |   6.000 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("cos", "π")]` |   1.800 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("cos", 0.0)]` |   6.400 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("sin", "π")]` |   1.800 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("sin", 0.0)]` |   5.500 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("tan", "π")]` |   1.800 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("tan", 0.0)]` |   6.300 ns (5%) |          |                 |             |
| `["utf8", "join"]`                             | 215.486 ms (5%) | 1.581 ms | 127.36 MiB (1%) |          21 |
| `["utf8", "replace"]`                          | 123.601 μs (5%) |          |  12.00 KiB (1%) |           4 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["trigonometry", "circular"]`
- `["trigonometry", "hyperbolic"]`
- `["utf8"]`

## Julia versioninfo
```
Julia Version 1.9.3
Commit bed2cd540a1 (2023-08-24 14:43 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 22.04.3 LTS
  uname: Linux 6.2.0-1015-azure #15~22.04.1-Ubuntu SMP Fri Oct  6 13:20:44 UTC 2023 x86_64 x86_64
  CPU: Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz: 
              speed         user         nice          sys         idle          irq
       #1  2294 MHz        798 s          0 s        147 s       2039 s          0 s
       #2  2294 MHz       1367 s          0 s        223 s       1381 s          0 s
  Memory: 6.759746551513672 GB (5434.89453125 MB free)
  Uptime: 307.41 sec
  Load Avg:  1.34  0.88  0.4
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, broadwell)
  Threads: 1 on 2 virtual cores
```

---
# Baseline result
# Benchmark Report for */home/runner/work/HMatrices.jl/HMatrices.jl*

## Job Properties
* Time of benchmark: 26 Oct 2023 - 6:48
* Package commit: 0132e0
* Julia commit: bed2cd
* Julia command flags: None
* Environment variables: None

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                             | time            | GC time  | memory          | allocations |
|------------------------------------------------|----------------:|---------:|----------------:|------------:|
| `["trigonometry", "circular", ("cos", "π")]`   |   1.600 ns (5%) |          |                 |             |
| `["trigonometry", "circular", ("cos", 0.0)]`   |   6.400 ns (5%) |          |                 |             |
| `["trigonometry", "circular", ("sin", "π")]`   |   1.700 ns (5%) |          |                 |             |
| `["trigonometry", "circular", ("sin", 0.0)]`   |   6.400 ns (5%) |          |                 |             |
| `["trigonometry", "circular", ("tan", "π")]`   |   1.600 ns (5%) |          |                 |             |
| `["trigonometry", "circular", ("tan", 0.0)]`   |   5.900 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("cos", "π")]` |   1.800 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("cos", 0.0)]` |   7.000 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("sin", "π")]` |   1.700 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("sin", 0.0)]` |   5.300 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("tan", "π")]` |   1.700 ns (5%) |          |                 |             |
| `["trigonometry", "hyperbolic", ("tan", 0.0)]` |   5.500 ns (5%) |          |                 |             |
| `["utf8", "join"]`                             | 220.653 ms (5%) | 1.764 ms | 127.36 MiB (1%) |          21 |
| `["utf8", "replace"]`                          | 125.401 μs (5%) |          |  12.00 KiB (1%) |           4 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["trigonometry", "circular"]`
- `["trigonometry", "hyperbolic"]`
- `["utf8"]`

## Julia versioninfo
```
Julia Version 1.9.3
Commit bed2cd540a1 (2023-08-24 14:43 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 22.04.3 LTS
  uname: Linux 6.2.0-1015-azure #15~22.04.1-Ubuntu SMP Fri Oct  6 13:20:44 UTC 2023 x86_64 x86_64
  CPU: Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz: 
              speed         user         nice          sys         idle          irq
       #1  2294 MHz       1083 s          0 s        176 s       2080 s          0 s
       #2  2294 MHz       1413 s          0 s        231 s       1680 s          0 s
  Memory: 6.759746551513672 GB (5377.80859375 MB free)
  Uptime: 342.93 sec
  Load Avg:  1.25  0.91  0.43
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, broadwell)
  Threads: 1 on 2 virtual cores
```

---
# Runtime information
| Runtime Info | |
|:--|:--|
| BLAS #threads | 1 |
| `BLAS.vendor()` | `lbt` |
| `Sys.CPU_THREADS` | 2 |

`lscpu` output:

    Architecture:                       x86_64
    CPU op-mode(s):                     32-bit, 64-bit
    Address sizes:                      46 bits physical, 48 bits virtual
    Byte Order:                         Little Endian
    CPU(s):                             2
    On-line CPU(s) list:                0,1
    Vendor ID:                          GenuineIntel
    Model name:                         Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz
    CPU family:                         6
    Model:                              79
    Thread(s) per core:                 1
    Core(s) per socket:                 2
    Socket(s):                          1
    Stepping:                           1
    BogoMIPS:                           4589.36
    Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology cpuid pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single pti fsgsbase bmi1 hle avx2 smep bmi2 erms invpcid rtm rdseed adx smap xsaveopt md_clear
    Hypervisor vendor:                  Microsoft
    Virtualization type:                full
    L1d cache:                          64 KiB (2 instances)
    L1i cache:                          64 KiB (2 instances)
    L2 cache:                           512 KiB (2 instances)
    L3 cache:                           50 MiB (1 instance)
    NUMA node(s):                       1
    NUMA node0 CPU(s):                  0,1
    Vulnerability Gather data sampling: Not affected
    Vulnerability Itlb multihit:        KVM: Mitigation: VMX unsupported
    Vulnerability L1tf:                 Mitigation; PTE Inversion
    Vulnerability Mds:                  Mitigation; Clear CPU buffers; SMT Host state unknown
    Vulnerability Meltdown:             Mitigation; PTI
    Vulnerability Mmio stale data:      Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
    Vulnerability Retbleed:             Not affected
    Vulnerability Spec rstack overflow: Not affected
    Vulnerability Spec store bypass:    Vulnerable
    Vulnerability Spectre v1:           Mitigation; usercopy/swapgs barriers and __user pointer sanitization
    Vulnerability Spectre v2:           Mitigation; Retpolines, STIBP disabled, RSB filling, PBRSB-eIBRS Not affected
    Vulnerability Srbds:                Not affected
    Vulnerability Tsx async abort:      Mitigation; Clear CPU buffers; SMT Host state unknown
    

| Cpu Property       | Value                                                   |
|:------------------ |:------------------------------------------------------- |
| Brand              | Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz               |
| Vendor             | :Intel                                                  |
| Architecture       | :Broadwell                                              |
| Model              | Family: 0x06, Model: 0x4f, Stepping: 0x01, Type: 0x00   |
| Cores              | 2 physical cores, 2 logical cores (on executing CPU)    |
|                    | No Hyperthreading hardware capability detected          |
| Clock Frequencies  | Not supported by CPU                                    |
| Data Cache         | Level 1:3 : (32, 256, 51200) kbytes                     |
|                    | 64 byte cache line size                                 |
| Address Size       | 48 bits virtual, 46 bits physical                       |
| SIMD               | 256 bit = 32 byte max. SIMD vector size                 |
| Time Stamp Counter | TSC is accessible via `rdtsc`                           |
|                    | TSC increased at every clock cycle (non-invariant TSC)  |
| Perf. Monitoring   | Performance Monitoring Counters (PMC) are not supported |
| Hypervisor         | Yes, Microsoft                                          |

