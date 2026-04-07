# SCALE-Sim Neutron Integration Notes

This document summarizes the discussion and follow-up work around running SCALE-Sim testcases and reviewing the Neutron dataflow integration

## 1. Repository and environment overview

The repository contains the main SCALE-Sim code under `scalesim/`, configuration files under `configs/`, workload descriptions under `topologies/`, layout files under `layouts/`, regression assets under `test/`, and a DRAM submodule under `submodules/ramulator/`

Main entry points:

- Installed package mode
  - `python3 -m scalesim.scale -c <config> -t <topology> -p <output_dir>`
- Source mode
  - `PYTHONPATH=. python3 ./scalesim/scale.py -c <config> -t <topology> -p <output_dir>`

Observed local environment during review:

- System Python was available
- The repository `.venv` was usable for execution
- Core Python dependencies such as `numpy`, `pandas`, `tqdm`, `numba`, `matplotlib`, and `absl` could be imported
- `matplotlib` preferred `MPLCONFIGDIR=/tmp/matplotlib` because the default config path was not writable

## 2. How to run a testcase from the repository

A working convolution testcase:

```bash
cd /home/zhuozi/SCALE-Sim
MPLCONFIGDIR=/tmp/matplotlib ./.venv/bin/python -m scalesim.scale \
  -c configs/scale.cfg \
  -t topologies/conv_nets/alexnet_part.csv \
  -p /tmp/scalesim_try3
```

A working GEMM testcase:

```bash
cd /home/zhuozi/SCALE-Sim
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. ./.venv/bin/python ./scalesim/scale.py \
  -c configs/scale.cfg \
  -t topologies/GEMM_mnk/vit_s.csv \
  -i gemm \
  -p /tmp/scalesim_vit
```

Output files are generated under:

- `COMPUTE_REPORT.csv`
- `BANDWIDTH_REPORT.csv`
- `DETAILED_ACCESS_REPORT.csv`
- `TIME_REPORT.csv`
- Layer-wise SRAM and DRAM traces under `layer*/`

## 3. Files involved in Neutron integration

The Neutron-related changes were concentrated in these files:

- `scalesim/compute/neutron_compute.py`
- `scalesim/single_layer_sim.py`
- `scalesim/scale_config.py`
- `scalesim/topology_utils.py`
- `configs/neutron.cfg`
- `scalesim/transfer_neutron/neutron.py`
- `scalesim/transfer_neutron/model_based_benchmarks.py`

Additional debug-oriented edits had also appeared in:

- `scalesim/compute/operand_matrix.py`
- `scalesim/memory/read_buffer.py`
- `scalesim/memory/double_buffered_scratchpad_mem.py`

## 4. Main issues found in the original Neutron integration

### 4.1 The Neutron path did not run

The original Neutron integration failed at runtime because `scalesim/compute/neutron_compute.py` defined two classes with the same name `neutron_compute`

- One class was the analytical Neutron model
- The second class was the SCALE-Sim adapter

Inside the adapter, `neutron_compute(arr_row, arr_col, num_accums)` tried to instantiate the model, but the adapter class had already overwritten the model class name, causing a `TypeError`

### 4.2 Dataflow naming was inconsistent

Different parts of the code used different names for the same concept:

- `nt_ds`
- `neutron`

This caused:

- warnings during config parsing
- incorrect banner output
- broken `TIME_REPORT` lookup and spatiotemporal routing

### 4.3 The accumulator count was not configurable

The original Neutron adapter derived `num_accums` from `arr_row * arr_col`, which prevented testing different accumulator-capacity design points independently of array shape

### 4.4 Debug prints polluted normal runs

Several debug prints were active in operand and memory code paths, making normal reports noisy and unsuitable for regression comparison

### 4.5 The current memory demand generation is still approximate

The current adapter does not yet generate true cycle-by-cycle Neutron operand scheduling

Instead, it:

1. Runs the analytical Neutron model
2. Obtains total counters such as total cycles and total operand reads and writes
3. Builds a synthetic demand stream from those totals so that the Scale-Sim memory backend can consume it

This is useful for integrating Neutron into the SCALE-Sim flow, but it is not yet a faithful trace of real Neutron pipeline scheduling

## 5. What was changed to make the Neutron path runnable

The following fixes were applied:

- Renamed the analytical model class so it no longer conflicts with the adapter
- Unified the main runtime dataflow name to `neutron`
- Added backward handling for the legacy alias `nt_ds`
- Added a dedicated `NeutronNumAccums` configuration parameter
- Updated the run banner to print `Neutron`
- Removed active debug prints that polluted standard output

Relevant updated files:

- `scalesim/compute/neutron_compute.py`
- `scalesim/scale_config.py`
- `scalesim/single_layer_sim.py`
- `scalesim/scale_sim.py`
- `configs/neutron.cfg`
- `scalesim/compute/operand_matrix.py`
- `scalesim/memory/double_buffered_scratchpad_mem.py`

## 6. Working Neutron testcase after the fixes

The following command ran successfully:

```bash
cd /home/zhuozi/SCALE-Sim
MPLCONFIGDIR=/tmp/matplotlib ./.venv/bin/python -m scalesim.scale \
  -c configs/neutron.cfg \
  -t topologies/GEMM_mnk/vit_s.csv \
  -i gemm \
  -p /tmp/neutron_run
```

Generated output directory:

- `/tmp/neutron_run/Neutron_NPU`

Generated files included:

- `COMPUTE_REPORT.csv`
- `BANDWIDTH_REPORT.csv`
- `DETAILED_ACCESS_REPORT.csv`
- `TIME_REPORT.csv`
- layer-wise SRAM and DRAM traces

## 7. Meaning of synthesized demand from aggregate counters

The phrase "synthesized from aggregate counters" means the current Neutron adapter does not start from a true per-cycle execution trace

Instead, it first computes aggregate totals from the analytical model, for example:

- total clock cycles
- total ifmap reads
- total filter reads
- total ofmap writes

Then it constructs a demand matrix by spreading those totals across cycles

Conceptually:

- If the model says there were 100000 reads over 1000 cycles
- The adapter builds a request stream that roughly looks like 100 requests per cycle

This is done using:

- flattened address pools
- deduplicated operand addresses
- average bandwidth derived from `ceil(total_requests / total_cycles)`

So the current implementation is suitable for:

- total cycles
- total request counts
- rough average bandwidth studies

But it is not yet suitable for:

- accurate burst shape
- instantaneous bandwidth shape
- bank conflict realism
- true stall distribution
- faithful cycle-level Neutron trace validation

In short:

The current Neutron path is using Neutron totals to build a memory-compatible approximation, not replaying a real Neutron execution schedule

## 8. Remaining limitation

The Neutron path is now runnable, but the memory access pattern is still synthesized from aggregate counters rather than generated from a true cycle-accurate Neutron schedule

That means it is appropriate for integration testing and coarse report generation, but not yet for high-fidelity validation of Neutron memory behavior
