"""Benchmark circuit generators for temporal scheduling experiments."""
import networkx as nx
from qiskit.circuit import QuantumCircuit
import numpy as np


def make_qaoa_maxcut(n_qubits: int, seed: int = 42, depth: int = 1) -> QuantumCircuit:
    """QAOA circuit for Max-Cut on a random 3-regular graph."""
    G = nx.random_regular_graph(3, n_qubits, seed=seed)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    gamma, beta = 0.5, 0.3
    for _ in range(depth):
        for u, v in G.edges():
            qc.cx(u, v)
            qc.rz(2 * gamma, v)
            qc.cx(u, v)
        for i in range(n_qubits):
            qc.rx(2 * beta, i)
    return qc


def make_qft(n_qubits: int) -> QuantumCircuit:
    """Standard Quantum Fourier Transform circuit."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
        for j in range(i + 1, n_qubits):
            qc.cp(np.pi / 2 ** (j - i), i, j)
    return qc


def make_random_entanglement(n_qubits: int, depth: int = 10, seed: int = 42) -> QuantumCircuit:
    """Random entanglement circuit: D layers of randomly paired CX gates."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for _ in range(depth):
        perm = rng.permutation(n_qubits)
        for k in range(0, n_qubits - 1, 2):
            qc.cx(int(perm[k]), int(perm[k + 1]))
    return qc


def _phase_blocks(n_qubits: int, phase_width: int):
    if n_qubits % phase_width != 0:
        raise ValueError('n_qubits must be divisible by phase_width')
    return [list(range(i, i + phase_width)) for i in range(0, n_qubits, phase_width)]


def _round_robin(flat_groups):
    out = []
    max_len = max(len(g) for g in flat_groups)
    for idx in range(max_len):
        for g in flat_groups:
            if idx < len(g):
                out.append(g[idx])
    return out


def make_phase_order_circuit(
    n_qubits: int,
    phase_width: int = 4,
    repeats: int = 3,
    ordering: str = 'clustered',
) -> QuantumCircuit:
    """Same gate multiset, different ordering.

    ordering='clustered' keeps interactions phase-local.
    ordering='interleaved' round-robins interactions across phases.
    Both orders use the same two-qubit gate multiset.
    """
    blocks = _phase_blocks(n_qubits, phase_width)
    per_phase_sequences = []
    for rep in range(repeats):
        for block_idx, block in enumerate(blocks):
            seq = []
            for i in range(len(block) - 1):
                seq.append((block[i], block[i + 1]))
            if rep % 2 == 1:
                seq = list(reversed(seq))
            if block_idx < len(blocks) - 1:
                seq.append((block[-1], blocks[block_idx + 1][0]))
            per_phase_sequences.append(seq)

    if ordering == 'clustered':
        ordered_pairs = [pair for seq in per_phase_sequences for pair in seq]
    elif ordering == 'interleaved':
        ordered_pairs = _round_robin(per_phase_sequences)
    else:
        raise ValueError(f'Unknown ordering: {ordering}')

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    for a, b in ordered_pairs:
        qc.cx(a, b)
    return qc


def make_same_graph_order_pair(n_qubits: int = 12, phase_width: int = 4, repeats: int = 3):
    """Return a pair of circuits with the same two-qubit gate multiset but different order."""
    return (
        make_phase_order_circuit(n_qubits, phase_width=phase_width, repeats=repeats, ordering='clustered'),
        make_phase_order_circuit(n_qubits, phase_width=phase_width, repeats=repeats, ordering='interleaved'),
    )


def make_ancilla_reuse_workload(
    n_data_qubits: int = 12,
    n_ancillas: int = 2,
    phase_size: int = 4,
    rounds: int = 2,
) -> QuantumCircuit:
    """Dynamic-circuit-inspired workload with ancilla reuse and reset-like phases."""
    total_qubits = n_data_qubits + n_ancillas
    qc = QuantumCircuit(total_qubits)
    ancillas = list(range(n_data_qubits, total_qubits))
    groups = [list(range(i, min(i + phase_size, n_data_qubits))) for i in range(0, n_data_qubits, phase_size)]

    for q in range(n_data_qubits):
        qc.h(q)

    for r in range(rounds):
        for group_idx, group in enumerate(groups):
            anc = ancillas[group_idx % n_ancillas]
            qc.reset(anc)
            qc.h(anc)
            for q in group:
                qc.cx(anc, q)
            for q in reversed(group):
                qc.cz(q, anc)
            if r < rounds - 1:
                qc.barrier()
    return qc


def make_working_set_shift_circuit(
    n_qubits: int = 16,
    window: int = 4,
    phases: int = 4,
) -> QuantumCircuit:
    """Synthetic workload with explicit active-set shifts across phases."""
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    starts = np.linspace(0, max(0, n_qubits - window), num=phases, dtype=int)
    for p, start in enumerate(starts):
        active = list(range(int(start), min(int(start) + window, n_qubits)))
        for i in range(len(active) - 1):
            qc.cx(active[i], active[i + 1])
        if p < len(starts) - 1:
            next_start = int(starts[p + 1])
            bridge_target = next_start if next_start != active[-1] else min(next_start + 1, n_qubits - 1)
            if bridge_target != active[-1]:
                qc.cx(active[-1], bridge_target)
        qc.barrier()
    return qc


def get_all_benchmarks(sizes=(22, 24, 26, 28), qaoa_seeds=(42, 43, 44, 45, 46),
                       random_seeds=(42, 43, 44, 45, 46)):
    """Generate all benchmark circuits as (name, circuit) pairs."""
    benchmarks = []
    for n in sizes:
        for s in qaoa_seeds:
            benchmarks.append((f"QAOA_{n}q_s{s}", make_qaoa_maxcut(n, seed=s)))
        benchmarks.append((f"QFT_{n}q", make_qft(n)))
        for s in random_seeds:
            benchmarks.append((f"Random_{n}q_s{s}", make_random_entanglement(n, seed=s)))
    return benchmarks


if __name__ == '__main__':
    for name, qc in get_all_benchmarks(sizes=(22,), qaoa_seeds=(42,), random_seeds=(42,)):
        n2q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
        print(f"{name}: {qc.num_qubits}q, {n2q} two-qubit gates")
    c1, c2 = make_same_graph_order_pair()
    print('same-graph pair two-qubit gates:', sum(1 for inst in c1.data if inst.operation.num_qubits == 2), sum(1 for inst in c2.data if inst.operation.num_qubits == 2))
