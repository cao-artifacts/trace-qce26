"""
Experiment 5: Scalability — scheduler runtime at large circuit scales.
Also implements a simple ILP baseline for comparison.
"""
import json
import os
import time
from circuits import make_qaoa_maxcut
from qvm_scheduler import schedule

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_scheduler_scalability():
    """Run LR-Gate and Lookahead on increasing circuit sizes."""
    print("=" * 60)
    print("SCALABILITY: Scheduler Runtime")
    print("=" * 60)

    rows = []
    configs = [
        (28, 20), (50, 40), (100, 80), (200, 160),
        (500, 400), (1000, 800),
    ]

    for N, M in configs:
        print(f"\n  N={N}, M={M}")
        qc = make_qaoa_maxcut(N, seed=42)
        n2q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)

        for policy in ["LR-Gate", "Lookahead"]:
            t0 = time.time()
            result = schedule(qc, M=M, policy=policy, W=15, seed=42)
            dt = time.time() - t0
            row = {
                "N": N, "M": M, "policy": policy,
                "K": result.K, "n_2q_gates": n2q,
                "sched_time_s": round(dt, 4),
            }
            rows.append(row)
            print(f"    {policy:>10}: K={result.K}, time={dt:.4f}s")

    outpath = os.path.join(RESULTS_DIR, "exp5_scalability.json")
    with open(outpath, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} rows to {outpath}")


def run_ilp_baseline():
    """
    Simple ILP for min-cut qubit partitioning.
    Uses PuLP to find minimum edge cuts in the qubit interaction graph
    that partition qubits into groups of size <= M.
    """
    try:
        import pulp
    except ImportError:
        print("Installing PuLP...")
        os.system("pip install pulp -q")
        import pulp

    print("\n" + "=" * 60)
    print("ILP BASELINE: Min-Cut Graph Partitioning")
    print("=" * 60)

    rows = []
    configs = [
        (28, 20), (50, 40), (100, 80), (200, 160),
    ]
    TIMEOUT = 120  # seconds

    for N, M in configs:
        print(f"\n  N={N}, M={M}")
        qc = make_qaoa_maxcut(N, seed=42)

        # Build interaction graph: edges between qubits that share a 2q gate
        edges = set()
        for inst in qc.data:
            qubits = [qc.find_bit(q).index for q in inst.qubits]
            if len(qubits) == 2:
                edges.add((min(qubits), max(qubits)))
        edges = list(edges)

        # Number of partitions needed
        n_parts = (N + M - 1) // M  # ceil(N/M)

        # ILP: assign each qubit to a partition, minimize cross-partition edges
        prob = pulp.LpProblem("MinCutPartition", pulp.LpMinimize)

        # x[i][p] = 1 if qubit i is in partition p
        x = {}
        for i in range(N):
            for p in range(n_parts):
                x[i, p] = pulp.LpVariable(f"x_{i}_{p}", cat="Binary")

        # y[e] = 1 if edge e crosses partitions
        y = {}
        for idx, (u, v) in enumerate(edges):
            y[idx] = pulp.LpVariable(f"y_{idx}", cat="Binary")

        # Objective: minimize cross-partition edges
        prob += pulp.lpSum(y[idx] for idx in range(len(edges)))

        # Each qubit in exactly one partition
        for i in range(N):
            prob += pulp.lpSum(x[i, p] for p in range(n_parts)) == 1

        # Partition size <= M
        for p in range(n_parts):
            prob += pulp.lpSum(x[i, p] for i in range(N)) <= M

        # Edge crossing: y[e] >= x[u][p] - x[v][p] for all p
        for idx, (u, v) in enumerate(edges):
            for p in range(n_parts):
                prob += y[idx] >= x[u, p] - x[v, p]
                prob += y[idx] >= x[v, p] - x[u, p]

        # Solve with timeout
        t0 = time.time()
        solver = pulp.PULP_CBC_CMD(timeLimit=TIMEOUT, msg=0)
        status = prob.solve(solver)
        dt = time.time() - t0

        if pulp.LpStatus[status] in ("Optimal", "Not Solved"):
            K_ilp = int(pulp.value(prob.objective)) if pulp.value(prob.objective) else None
            solved = pulp.LpStatus[status] == "Optimal"
        else:
            K_ilp = None
            solved = False

        # Compare with LR-Gate
        t1 = time.time()
        result_lr = schedule(qc, M=M, policy="LR-Gate", W=15, seed=42)
        dt_lr = time.time() - t1

        row = {
            "N": N, "M": M,
            "ilp_K": K_ilp, "ilp_time_s": round(dt, 2),
            "ilp_solved": solved, "ilp_status": pulp.LpStatus[status],
            "lrgate_K": result_lr.K, "lrgate_time_s": round(dt_lr, 4),
            "n_edges": len(edges), "n_parts": n_parts,
        }
        rows.append(row)
        status_str = f"K={K_ilp}" if solved else f"TIMEOUT ({TIMEOUT}s)"
        print(f"    ILP: {status_str}, time={dt:.1f}s")
        print(f"    LR-Gate: K={result_lr.K}, time={dt_lr:.4f}s")

    outpath = os.path.join(RESULTS_DIR, "exp5_ilp_baseline.json")
    with open(outpath, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} rows to {outpath}")


if __name__ == "__main__":
    run_scheduler_scalability()
    run_ilp_baseline()
    print("\n=== SCALABILITY EXPERIMENTS COMPLETE ===")
