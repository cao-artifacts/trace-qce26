"""Temporal-motivation experiments for the revised QVM paper."""
import hashlib
import json
import os
import time

from circuits import (
    make_qaoa_maxcut,
    make_random_entanglement,
    make_same_graph_order_pair,
    make_ancilla_reuse_workload,
    make_working_set_shift_circuit,
)
from qvm_scheduler import schedule
from temporal_oracle import (
    RandomPolicy,
    LRUPolicy,
    LRGatePolicy,
    LookaheadPolicy,
    exact_optimum_cut_count,
    extract_two_qubit_stream,
    first_m_distinct_qubits,
    mean_forward_reuse_distance,
    qiskit_to_gate_stream,
    schedule_gate_stream,
    working_set_turnover,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
POLICIES = ['Random', 'LRU', 'LR-Gate', 'Lookahead']


def _graph_signature(twoq_stream):
    edges = sorted({tuple(sorted((a, b))) for a, b in twoq_stream})
    payload = ';'.join(f'{a}-{b}' for a, b in edges)
    return hashlib.md5(payload.encode()).hexdigest()[:10], len(edges)


def run_exp6_order_sensitivity():
    print('=' * 60)
    print('EXPERIMENT 6: ORDER SENSITIVITY')
    print('=' * 60)
    rows = []
    configs = [
        (12, 4, 3, 8),
        (16, 4, 3, 12),
        (20, 4, 3, 16),
    ]
    for n, phase_width, repeats, m in configs:
        clustered, interleaved = make_same_graph_order_pair(n, phase_width=phase_width, repeats=repeats)
        for ordering, qc in [('clustered', clustered), ('interleaved', interleaved)]:
            twoq = extract_two_qubit_stream(qiskit_to_gate_stream(qc))
            sig, n_edges = _graph_signature(twoq)
            for policy in POLICIES:
                t0 = time.time()
                result = schedule(qc, M=m, policy=policy, W=15, seed=42)
                dt = time.time() - t0
                rows.append({
                    'family': 'same_graph_order_pair',
                    'ordering': ordering,
                    'N': n,
                    'M': m,
                    'phase_width': phase_width,
                    'repeats': repeats,
                    'graph_signature': sig,
                    'n_static_edges': n_edges,
                    'policy': policy,
                    'K': result.K,
                    'sched_time_s': round(dt, 5),
                    'mean_reuse_distance': round(mean_forward_reuse_distance(twoq), 4),
                    'working_set_turnover': round(working_set_turnover(twoq), 4),
                })
                print(f'  N={n} {ordering:11s} {policy:>10s}: K={result.K}')
    outpath = os.path.join(RESULTS_DIR, 'exp_rq1_order_sensitivity.json')
    with open(outpath, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'Saved {len(rows)} rows to {outpath}')
    return rows


def run_exp7_oracle_gap():
    print('\n' + '=' * 60)
    print('EXPERIMENT 7: EXACT ORACLE GAP')
    print('=' * 60)
    rows = []
    workloads = []
    for depth in (1, 2):
        workloads.append((f'qaoa_p{depth}', make_qaoa_maxcut(8, seed=42 + depth, depth=depth), 5))
    clustered, interleaved = make_same_graph_order_pair(8, phase_width=4, repeats=2)
    workloads.extend([
        ('phase_clustered', clustered, 5),
        ('phase_interleaved', interleaved, 5),
        ('ancilla_reuse', make_ancilla_reuse_workload(n_data_qubits=6, n_ancillas=2, phase_size=3, rounds=2), 5),
        ('working_set_shift', make_working_set_shift_circuit(n_qubits=8, window=4, phases=3), 5),
        ('random_seed42', make_random_entanglement(8, depth=5, seed=42), 5),
        ('random_seed43', make_random_entanglement(8, depth=5, seed=43), 5),
    ])

    policy_builders = {
        'Random': lambda: RandomPolicy(seed=0),
        'LRU': LRUPolicy,
        'LR-Gate': lambda: LRGatePolicy(window=4),
        'Lookahead': LookaheadPolicy,
    }

    for name, qc, m in workloads:
        gates = qiskit_to_gate_stream(qc)
        twoq = extract_two_qubit_stream(gates)
        initial = first_m_distinct_qubits(gates, m)
        optimum = exact_optimum_cut_count(twoq, m, initial_resident=initial)
        mean_reuse = mean_forward_reuse_distance(twoq)
        turnover = working_set_turnover(twoq)
        print(f'  workload={name:18s} OPT={optimum}')
        for policy_name, builder in policy_builders.items():
            policy = builder() if callable(builder) else builder()
            cuts = schedule_gate_stream(gates, m, policy, initial_resident=initial)
            rows.append({
                'workload': name,
                'N': qc.num_qubits,
                'M': m,
                'policy': policy_name,
                'K': cuts,
                'K_opt': optimum,
                'gap_abs': cuts - optimum,
                'gap_ratio': round(cuts / optimum, 4) if optimum > 0 else 1.0,
                'mean_reuse_distance': round(mean_reuse, 4),
                'working_set_turnover': round(turnover, 4),
            })
            print(f'    {policy_name:10s} K={cuts} gap={cuts - optimum:+d}')
    outpath = os.path.join(RESULTS_DIR, 'exp_rq2_oracle_gap.json')
    with open(outpath, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'Saved {len(rows)} rows to {outpath}')
    return rows


def run_exp8_dynamic_workloads():
    print('\n' + '=' * 60)
    print('EXPERIMENT 8: DYNAMIC-CIRCUIT-INSPIRED WORKLOADS')
    print('=' * 60)
    rows = []
    workloads = [
        ('ancilla_reuse_r2', make_ancilla_reuse_workload(n_data_qubits=12, n_ancillas=2, phase_size=4, rounds=2), 10),
        ('ancilla_reuse_r3', make_ancilla_reuse_workload(n_data_qubits=12, n_ancillas=2, phase_size=4, rounds=3), 10),
        ('working_set_shift_p4', make_working_set_shift_circuit(n_qubits=16, window=4, phases=4), 12),
        ('working_set_shift_p6', make_working_set_shift_circuit(n_qubits=16, window=4, phases=6), 12),
        ('qaoa_p3_16q', make_qaoa_maxcut(16, seed=42, depth=3), 12),
    ]
    for name, qc, m in workloads:
        twoq = extract_two_qubit_stream(qiskit_to_gate_stream(qc))
        sig, n_edges = _graph_signature(twoq)
        mean_reuse = mean_forward_reuse_distance(twoq)
        turnover = working_set_turnover(twoq)
        print(f'  workload={name}')
        for policy in POLICIES:
            t0 = time.time()
            result = schedule(qc, M=m, policy=policy, W=15, seed=42)
            dt = time.time() - t0
            rows.append({
                'workload': name,
                'N': qc.num_qubits,
                'M': m,
                'policy': policy,
                'K': result.K,
                'sched_time_s': round(dt, 5),
                'graph_signature': sig,
                'n_static_edges': n_edges,
                'mean_reuse_distance': round(mean_reuse, 4),
                'working_set_turnover': round(turnover, 4),
                'n_two_qubit_gates': len(twoq),
            })
            print(f'    {policy:10s} K={result.K}')
    outpath = os.path.join(RESULTS_DIR, 'exp_rq4_temporal_workloads.json')
    with open(outpath, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'Saved {len(rows)} rows to {outpath}')
    return rows


if __name__ == '__main__':
    run_exp6_order_sensitivity()
    run_exp7_oracle_gap()
    run_exp8_dynamic_workloads()
    print('\n=== TEMPORAL EXPERIMENTS COMPLETE ===')
