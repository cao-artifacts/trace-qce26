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

This is a **minimal, self-contained artifact**: it contains the scheduler, the workload
generators, the exact oracle, and the drivers for the paper's main simulation experiments.
All experiments use **fixed random seeds**, so every result file regenerates
deterministically from the code below — no raw data is shipped. The end-to-end IBM Quantum
hardware runs reported in the paper are not included (they require an IBM Quantum account
and QPU time).

## Contents

| File | Role |
|---|---|
| `qvm_scheduler.py` | TRACE scheduler with the four eviction policies (Random, LRU, LR-Gate, Lookahead) |
| `circuits.py` | Workload generators (QAOA, QFT, random entanglement, temporal/ancilla-reuse families) |
| `temporal_oracle.py` | Exact dynamic-programming oracle (OPT) for small instances |
| `run_temporal_experiments.py` | Order sensitivity (RQ1), oracle gap (RQ2), dynamic-circuit-inspired workloads (RQ4) |
| `run_experiments.py` | Cut counts, capacity sweep, window ablation, baselines on QAOA/QFT/random workloads |
| `run_scalability.py` | Scheduler runtime / planning-cost scaling |
| `plot_temporal_results.py` | Regenerates the temporal-result figures from the JSONs above |

## Setup

Tested with Python 3.12, Qiskit 2.3.1, qiskit-addon-cutting 0.10.0, networkx 3.6.1,
numpy 2.4.4, matplotlib 3.10.8 (matplotlib/seaborn needed only for plotting):

```bash
pip install qiskit qiskit-addon-cutting qiskit-aer networkx numpy matplotlib seaborn
```

## Reproducing the results

Each driver writes its JSON outputs into `results/` (created on first run):

```bash
python run_temporal_experiments.py   # -> exp_rq1_order_sensitivity.json,
                                     #    exp_rq2_oracle_gap.json,
                                     #    exp_rq4_temporal_workloads.json
python run_experiments.py            # -> exp1_* cut-count / ablation / baseline JSONs
python run_scalability.py            # -> exp5_scalability.json
python plot_temporal_results.py      # -> figures from the exp_rq* JSONs
```

Seeds are fixed inside the drivers, so re-running reproduces the numbers reported in the
paper for these experiments.

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
