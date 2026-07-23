# TRACE: Temporal Residency-Aware Circuit Execution for Over-Capacity Quantum Hardware

Research artifact for the paper accepted at **IEEE QCE 2026** (IEEE International Conference
on Quantum Computing and Engineering / IEEE Quantum Week), Quantum System Software (QSYS)
track, paper **QSYS-564**. Toronto, Canada, September 13–18, 2026.

**Authors:** Charles Cao, Sergei V. Kalinin (University of Tennessee, Knoxville);
Jong Youl Choi, Ahmad Maroof Karimi (Oak Ridge National Laboratory).

TRACE reframes executing an $N$-qubit circuit on only $M < N$ physical qubits as a
*temporal qubit residency scheduling* problem over the gate stream: it maintains a resident
set of $M$ qubits and inserts a wire cut whenever a two-qubit gate references a non-resident
operand. Its main practical policy, **LR-Gate** (Latest-Reuse Gate), scans a bounded window
of upcoming gates and evicts the resident qubit whose next two-qubit-gate use lies latest.

This is a **minimal artifact**:

- `src/` contains the scheduler, workload generators, exact oracle, and the drivers for the
  paper's main simulation experiments. All simulation experiments use **fixed random
  seeds**, so their result files regenerate deterministically — no simulation data is
  shipped.
- `data/ibm-hardware/` contains the raw results of the paper's IBM Quantum hardware
  validation. Hardware runs are **not** deterministically reproducible (device noise,
  calibration drift, QPU access required), so this one-run snapshot is included as-is —
  see the disclaimer in [`data/ibm-hardware/README.md`](data/ibm-hardware/README.md).

## Contents

| Path | Role |
|---|---|
| `src/qvm_scheduler.py` | TRACE scheduler with the four eviction policies (Random, LRU, LR-Gate, Lookahead) |
| `src/circuits.py` | Workload generators (QAOA, QFT, random entanglement, temporal/ancilla-reuse families) |
| `src/temporal_oracle.py` | Exact dynamic-programming oracle (OPT) for small instances |
| `src/run_temporal_experiments.py` | Order sensitivity (RQ1), oracle gap (RQ2), dynamic-circuit-inspired workloads (RQ4) |
| `src/run_experiments.py` | Cut counts, capacity sweep, window ablation, baselines on QAOA/QFT/random workloads |
| `src/run_scalability.py` | Scheduler runtime / planning-cost scaling |
| `src/plot_temporal_results.py` | Regenerates the temporal-result figures from the JSONs above |
| `data/ibm-hardware/` | Single-run IBM Quantum (ibm_fez, 156 qubits) hardware results + disclaimer |

## Setup

Tested with Python 3.12, Qiskit 2.3.1, qiskit-addon-cutting 0.10.0, networkx 3.6.1,
numpy 2.4.4, matplotlib 3.10.8 (matplotlib/seaborn needed only for plotting):

```bash
pip install qiskit qiskit-addon-cutting qiskit-aer networkx numpy matplotlib seaborn
```

## Reproducing the simulation results

Each driver writes its JSON outputs into `src/results/` (next to the scripts, created on
first run — the paths work from any working directory):

```bash
python src/run_temporal_experiments.py   # -> src/results/exp_rq1_order_sensitivity.json,
                                         #    src/results/exp_rq2_oracle_gap.json,
                                         #    src/results/exp_rq4_temporal_workloads.json
python src/run_experiments.py            # -> src/results/exp1_* cut-count / ablation / baseline JSONs
python src/run_scalability.py            # -> src/results/exp5_scalability.json
python src/plot_temporal_results.py      # -> figures from the exp_rq* JSONs
```

Seeds are fixed inside the drivers, so re-running reproduces the numbers reported in the
paper for these experiments. The hardware numbers in `data/ibm-hardware/` are a one-time
snapshot and will not reproduce exactly (see its README).

## Citation

```bibtex
@inproceedings{cao2026trace,
  author    = {Cao, Charles and Kalinin, Sergei V. and Choi, Jong Youl and Karimi, Ahmad Maroof},
  title     = {{TRACE}: Temporal Residency-Aware Circuit Execution for Over-Capacity Quantum Hardware},
  booktitle = {IEEE International Conference on Quantum Computing and Engineering (QCE)},
  year      = {2026}
}
```

## License and status

MIT License. This repository is a research artifact archived as published; it is not
actively maintained.
