# IBM Quantum hardware data (single-run snapshot)

Raw results from the paper's end-to-end hardware validation on **ibm_fez** (156-qubit IBM
Quantum processor), collected on **2026-04-26**: 4096 shots, 20 samples per configuration,
two N=10 / M=8 workloads across the four scheduling policies, 18 jobs, ~358 seconds of
total QPU time.

**These files document one specific run; they are provided for transparency, not for exact
reproduction.** Unlike the simulation experiments in `src/` (fixed seeds, deterministic),
hardware results depend on device noise, calibration drift, transpilation, and queue
conditions at execution time — re-running the same experiments will produce different
numbers, and doing so requires an IBM Quantum account with QPU access. Readers should not
expect to obtain identical values.

| File | Content |
|---|---|
| `ibm_hardware_results.json` | Per-workload, per-policy hardware execution results |
| `all_jobs_summary.json` | Per-job metadata: backend, sub-experiment counts, status, counts summaries, QPU usage |

Account-specific identifiers (IBM Cloud instance CRN, plan id) are `REDACTED`.
