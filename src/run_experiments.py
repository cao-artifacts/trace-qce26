"""
QVM Experiment Runner.
Outputs JSON data to results/ directory.
Figures are generated separately by plot_results.py.
"""
import json
import os
import time
from circuits import make_qaoa_maxcut, make_qft, make_random_entanglement
from qvm_scheduler import schedule

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

POLICIES = ["Random", "LRU", "LR-Gate", "Lookahead"]
SIZES = [22, 24, 26, 28]
QAOA_SEEDS = [42, 43, 44, 45, 46]
RANDOM_SEEDS = [42, 43, 44, 45, 46]
M_DEFAULT = 20


def run_exp1_cut_count():
    """Experiment 1: Cut count comparison across policies, circuits, sizes."""
    print("=" * 60)
    print("EXPERIMENT 1: Cut Count Comparison")
    print("=" * 60)

    rows = []

    # QAOA
    for n in SIZES:
        for seed in QAOA_SEEDS:
            qc = make_qaoa_maxcut(n, seed=seed)
            n2q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
            for policy in POLICIES:
                t0 = time.time()
                result = schedule(qc, M=M_DEFAULT, policy=policy, seed=seed)
                dt = time.time() - t0
                row = {
                    "circuit": "QAOA", "N": n, "seed": seed,
                    "policy": policy, "K": result.K,
                    "overhead_4K": 4 ** result.K,
                    "n_2q_gates": n2q,
                    "sched_time_s": round(dt, 4),
                }
                rows.append(row)
                print(f"  QAOA N={n} seed={seed} {policy:>10}: K={result.K}")

    # QFT
    for n in SIZES:
        qc = make_qft(n)
        n2q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
        for policy in POLICIES:
            t0 = time.time()
            result = schedule(qc, M=M_DEFAULT, policy=policy)
            dt = time.time() - t0
            row = {
                "circuit": "QFT", "N": n, "seed": 0,
                "policy": policy, "K": result.K,
                "overhead_4K": 4 ** result.K,
                "n_2q_gates": n2q,
                "sched_time_s": round(dt, 4),
            }
            rows.append(row)
            print(f"  QFT  N={n}          {policy:>10}: K={result.K}")

    # Random
    for n in SIZES:
        for seed in RANDOM_SEEDS:
            qc = make_random_entanglement(n, seed=seed)
            n2q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
            for policy in POLICIES:
                t0 = time.time()
                result = schedule(qc, M=M_DEFAULT, policy=policy, seed=seed)
                dt = time.time() - t0
                row = {
                    "circuit": "Random", "N": n, "seed": seed,
                    "policy": policy, "K": result.K,
                    "overhead_4K": 4 ** result.K,
                    "n_2q_gates": n2q,
                    "sched_time_s": round(dt, 4),
                }
                rows.append(row)
                print(f"  Rand N={n} seed={seed} {policy:>10}: K={result.K}")

    outpath = os.path.join(RESULTS_DIR, "exp1_cut_count.json")
    with open(outpath, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} rows to {outpath}")
    return rows


def run_exp1b_vary_M():
    """Experiment 1b: Sensitivity to physical capacity M."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1b: Vary M")
    print("=" * 60)

    rows = []
    N_FIXED = 24
    M_VALUES = [12, 14, 16, 18, 20]

    for seed in QAOA_SEEDS:
        qc = make_qaoa_maxcut(N_FIXED, seed=seed)
        for m in M_VALUES:
            for policy in POLICIES:
                result = schedule(qc, M=m, policy=policy, seed=seed)
                row = {
                    "circuit": "QAOA", "N": N_FIXED, "M": m,
                    "seed": seed, "policy": policy, "K": result.K,
                }
                rows.append(row)
                print(f"  QAOA N={N_FIXED} M={m} seed={seed} {policy:>10}: K={result.K}")

    outpath = os.path.join(RESULTS_DIR, "exp1b_vary_M.json")
    with open(outpath, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} rows to {outpath}")
    return rows


def run_exp1c_window_ablation():
    """Experiment 1c: LR-Gate window size ablation."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1c: Window Ablation")
    print("=" * 60)

    rows = []
    N_FIXED = 24
    W_VALUES = [5, 10, 15, 20, 30, 50, 100]

    for seed in QAOA_SEEDS:
        qc = make_qaoa_maxcut(N_FIXED, seed=seed)
        for w in W_VALUES:
            result = schedule(qc, M=M_DEFAULT, policy="LR-Gate", W=w, seed=seed)
            row = {
                "circuit": "QAOA", "N": N_FIXED, "M": M_DEFAULT,
                "W": w, "seed": seed, "K": result.K,
            }
            rows.append(row)
            print(f"  QAOA N={N_FIXED} W={w} seed={seed}: K={result.K}")

    outpath = os.path.join(RESULTS_DIR, "exp1c_window_ablation.json")
    with open(outpath, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} rows to {outpath}")
    return rows


def run_exp1d_baseline():
    """Experiment 1d: Time-unaware fixed-window baseline."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1d: Fixed-Window Baseline")
    print("=" * 60)

    rows = []

    def fixed_window_cuts(qc, M):
        """Naive baseline: cut every time a new qubit beyond M is encountered."""
        # Simple greedy: scan gates, track active qubits in a window,
        # cut whenever we exceed M
        active = set()
        K = 0
        for inst in qc.data:
            qubits = [qc.find_bit(q).index for q in inst.qubits]
            if len(qubits) == 2:
                for q in qubits:
                    active.add(q)
                if len(active) > M:
                    # Evict oldest-added qubit
                    K += 1
                    active = set(list(active)[-M:])
        return K

    for n in SIZES:
        for seed in QAOA_SEEDS:
            qc = make_qaoa_maxcut(n, seed=seed)
            K_baseline = fixed_window_cuts(qc, M_DEFAULT)
            result_lrgate = schedule(qc, M=M_DEFAULT, policy="LR-Gate", seed=seed)
            result_look = schedule(qc, M=M_DEFAULT, policy="Lookahead", seed=seed)
            rows.append({
                "circuit": "QAOA", "N": n, "seed": seed,
                "K_fixed_window": K_baseline,
                "K_LR_Gate": result_lrgate.K,
                "K_Lookahead": result_look.K,
            })

        qc = make_qft(n)
        K_baseline = fixed_window_cuts(qc, M_DEFAULT)
        result_lrgate = schedule(qc, M=M_DEFAULT, policy="LR-Gate")
        result_look = schedule(qc, M=M_DEFAULT, policy="Lookahead")
        rows.append({
            "circuit": "QFT", "N": n, "seed": 0,
            "K_fixed_window": K_baseline,
            "K_LR_Gate": result_lrgate.K,
            "K_Lookahead": result_look.K,
        })

    outpath = os.path.join(RESULTS_DIR, "exp1d_baseline.json")
    with open(outpath, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} rows to {outpath}")
    return rows


if __name__ == "__main__":
    run_exp1_cut_count()
    run_exp1b_vary_M()
    run_exp1c_window_ablation()
    run_exp1d_baseline()
    print("\n=== ALL SCHEDULING EXPERIMENTS COMPLETE ===")
