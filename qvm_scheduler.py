"""
QVM Scheduler: temporal qubit residency policies for circuit cutting.

Given a circuit with N > M qubits, walks the gate stream and records the
resident-set decisions that would keep two-qubit requests within capacity M.
The original annotated circuit is kept for backward compatibility; the
capacity-certified lowering below additionally emits enough wire cuts and
partition labels for qiskit-addon-cutting to produce fragments of width <= M.
"""
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Hashable

from qiskit.circuit import QuantumCircuit
from qiskit_addon_cutting.instructions import CutWire, Move
from qiskit_addon_cutting.qpd import TwoQubitQPDGate


@dataclass
class CutRecord:
    gate_index: int  # two-qubit gate ordinal before which the cut is planned
    wire: int        # virtual qubit that was cut (victim)
    swapped_in: int  # virtual qubit that was loaded
    original_index: int | None = None


@dataclass
class ScheduleEvent:
    """One two-qubit request in the temporal resident-set trace."""

    gate_ordinal: int
    original_index: int
    operands: tuple[int, int]
    resident_before: tuple[int, ...]
    missing: tuple[int, ...]
    loads: list[tuple[int, int]]  # (loaded_qubit, victim_qubit)
    resident_after: tuple[int, ...]


@dataclass
class LoweringResult:
    """Capacity-certified lowering artifact for qiskit-addon-cutting.

    ``annotated_circuit`` contains conservative ``CutWire`` markers at every
    partition-label transition needed by the trace lowering. ``cut_circuit`` is
    the corresponding QPD-Move representation consumed by
    ``qiskit_addon_cutting.partition_problem``. Each non-``None`` label names
    one executable fragment; the lowering is capacity certified when every
    label appears at most ``M`` times and qiskit-addon-cutting reports
    subcircuits of width at most ``M``.
    """

    annotated_circuit: QuantumCircuit
    cut_circuit: QuantumCircuit
    partition_labels: list[Hashable | None]
    interface_count: int
    fragment_widths: dict[Hashable, int]
    max_fragment_width: int
    op_labels: dict[int, Hashable]


@dataclass
class ScheduleResult:
    policy: str
    circuit_name: str
    N: int
    M: int
    K: int
    cuts: list[CutRecord]
    annotated_circuit: QuantumCircuit
    extra: dict = field(default_factory=dict)
    trace: list[ScheduleEvent] = field(default_factory=list)


def _get_two_qubit_gates(qc: QuantumCircuit):
    """Extract gate list as (index, gate_name, [qubit_indices])."""
    gates = []
    for i, inst in enumerate(qc.data):
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        gates.append((i, inst.operation.name, qubits))
    return gates


def schedule(qc: QuantumCircuit, M: int, policy: str = "LR-Gate",
             W: int = 15, seed: int = 42) -> ScheduleResult:
    """
    Run the QVM scheduler on circuit qc with physical capacity M.

    Returns a ScheduleResult with the annotated circuit and cut records.
    """
    N = qc.num_qubits
    assert N > M, f"Circuit has {N} qubits, capacity is {M} — no scheduling needed"

    gates = _get_two_qubit_gates(qc)
    # Build future-use index for Lookahead / LR-Gate
    # future_2q[gate_idx] = list of (qubit, next_gate_idx) for 2-qubit gates from gate_idx onward
    future_use = {}  # qubit -> sorted list of gate indices where it appears in a 2q gate
    for idx, name, qubits in gates:
        if len(qubits) == 2:
            for q in qubits:
                future_use.setdefault(q, []).append(idx)

    # Initialize resident set: first M qubits encountered in gate order
    resident = set()
    for _, _, qubits in gates:
        for q in qubits:
            if q not in resident and len(resident) < M:
                resident.add(q)
            if len(resident) == M:
                break
        if len(resident) == M:
            break
    # If not enough qubits encountered, fill with remaining
    if len(resident) < M:
        for q in range(N):
            if q not in resident:
                resident.add(q)
            if len(resident) == M:
                break

    # LRU tracking
    lru_order = OrderedDict()
    for q in resident:
        lru_order[q] = 0

    rng = random.Random(seed)
    cuts = []
    trace = []
    twoq_ordinal = 0

    # Process gates
    for _, (orig_idx, gate_name, qubits) in enumerate(gates):
        if len(qubits) != 2:
            continue

        a, b = qubits[0], qubits[1]
        resident_before = tuple(sorted(resident))
        missing = [q for q in [a, b] if q not in resident]
        loads = []

        for q in missing:
            # Select victim
            protected = {a, b}
            victim = _select_victim(
                policy, resident, protected, orig_idx, gates, future_use,
                lru_order, W, M, rng
            )

            cuts.append(CutRecord(
                gate_index=twoq_ordinal,
                wire=victim,
                swapped_in=q,
                original_index=orig_idx,
            ))
            loads.append((q, victim))
            resident.discard(victim)
            resident.add(q)

            # Update LRU
            if victim in lru_order:
                del lru_order[victim]
            lru_order[q] = orig_idx

        # Update LRU for both operands
        for q in [a, b]:
            if q in lru_order:
                lru_order.move_to_end(q)
                lru_order[q] = orig_idx

        trace.append(ScheduleEvent(
            gate_ordinal=twoq_ordinal,
            original_index=orig_idx,
            operands=(a, b),
            resident_before=resident_before,
            missing=tuple(missing),
            loads=loads,
            resident_after=tuple(sorted(resident)),
        ))
        twoq_ordinal += 1

    # Build annotated circuit
    annotated = _build_annotated_circuit(qc, cuts, gates)

    return ScheduleResult(
        policy=policy,
        circuit_name="",
        N=N,
        M=M,
        K=len(cuts),
        cuts=cuts,
        annotated_circuit=annotated,
        extra={"W": W, "seed": seed, "initial_resident": tuple(sorted(
            trace[0].resident_before if trace else resident
        ))},
        trace=trace,
    )


def _select_victim(policy, resident, protected, current_gate_idx, gates,
                    future_use, lru_order, W, M, rng):
    """Select a victim qubit to evict from resident set."""
    candidates = sorted(q for q in resident if q not in protected)
    assert candidates, "No eviction candidates — all residents are protected"

    if policy == "Random":
        return rng.choice(candidates)

    elif policy == "LRU":
        # Evict the least recently used
        for q in lru_order:
            if q in candidates:
                return q
        return candidates[0]

    elif policy == "LR-Gate":
        # Scan next W 2-qubit gates, evict candidate with farthest (or no) next use
        window_end = current_gate_idx + W
        best_victim = None
        best_next_use = -1

        for q in candidates:
            q_uses = future_use.get(q, [])
            # Find next use after current gate
            next_use = None
            for u in q_uses:
                if u > current_gate_idx:
                    next_use = u
                    break
            if next_use is None or next_use >= window_end:
                # No use within window — ideal victim
                return q
            if next_use > best_next_use:
                best_next_use = next_use
                best_victim = q

        return best_victim if best_victim else candidates[0]

    elif policy == "Lookahead":
        # Bélády's optimal: evict candidate with farthest next use
        best_victim = None
        best_next_use = -1

        for q in candidates:
            q_uses = future_use.get(q, [])
            next_use = None
            for u in q_uses:
                if u > current_gate_idx:
                    next_use = u
                    break
            if next_use is None:
                return q  # Never used again — optimal victim
            if next_use > best_next_use:
                best_next_use = next_use
                best_victim = q

        return best_victim if best_victim else candidates[0]

    else:
        raise ValueError(f"Unknown policy: {policy}")


def _build_annotated_circuit(original: QuantumCircuit, cuts: list[CutRecord],
                              gates: list) -> QuantumCircuit:
    """Build a new circuit with CutWire instructions inserted at cut positions."""
    N = original.num_qubits
    qc = QuantumCircuit(N)

    # Map gate indices to cuts that should happen BEFORE that gate
    cuts_before = {}
    for c in cuts:
        cuts_before.setdefault(c.gate_index, []).append(c)

    gate_counter = 0
    for i, inst in enumerate(original.data):
        qubits = [original.find_bit(q).index for q in inst.qubits]
        is_2q = len(qubits) == 2

        if is_2q:
            # Check if cuts should be inserted before this 2q gate
            if gate_counter in cuts_before:
                for cut in cuts_before[gate_counter]:
                    qc.append(CutWire(), [cut.wire])
            gate_counter += 1

        # Apply the original gate
        qc.append(inst.operation, qubits)

    return qc


def build_capacity_certified_lowering(
    original: QuantumCircuit,
    result: ScheduleResult,
) -> LoweringResult:
    """Build a conservative capacity-certified lowering for ``result``.

    The historical ``annotated_circuit`` inserts one ``CutWire`` at each
    scheduler eviction.  Qiskit's automatic partitioner may still group many
    resulting wire segments into a subcircuit wider than ``M``.  This lowering
    instead derives explicit partition labels from the resident-set trace.

    Each two-qubit gate is assigned to the current residency epoch after any
    required loads have occurred.  One-qubit gates on resident qubits share the
    current epoch; one-qubit gates on non-resident qubits receive singleton
    local labels.  Whenever a qubit's next operation belongs to a different
    label than its previous operation, a ``CutWire`` is inserted before that
    operation.  Thus every emitted fragment contains only qubits that are live
    together in one resident set, plus singleton local fragments.

    This is intentionally conservative: the emitted interface count can exceed
    the scheduler's miss count ``K`` because qubits that remain resident across
    epoch boundaries may need continuity cuts to keep static Qiskit partition
    labels within capacity.
    """
    if not result.trace:
        raise ValueError("ScheduleResult has no trace; run schedule() first")

    op_labels = _operation_partition_labels(original, result)
    annotated, segment_labels, interface_count = _insert_label_transition_cuts(
        original, op_labels
    )

    partition_labels: list[Hashable | None] = []
    for labels in segment_labels:
        if labels:
            partition_labels.extend(labels)
        else:
            partition_labels.append(None)

    cut_circuit = _expand_cut_wires_to_moves(annotated, segment_labels)
    if len(partition_labels) != cut_circuit.num_qubits:
        raise ValueError(
            "Internal lowering error: partition label count "
            f"{len(partition_labels)} != cut-circuit width "
            f"{cut_circuit.num_qubits}"
        )

    widths = Counter(label for label in partition_labels if label is not None)
    fragment_widths = dict(widths)
    max_fragment_width = max(widths.values(), default=0)
    if max_fragment_width > result.M:
        raise ValueError(
            f"Lowering violates capacity M={result.M}: max fragment width "
            f"{max_fragment_width}"
        )

    return LoweringResult(
        annotated_circuit=annotated,
        cut_circuit=cut_circuit,
        partition_labels=partition_labels,
        interface_count=interface_count,
        fragment_widths=fragment_widths,
        max_fragment_width=max_fragment_width,
        op_labels=op_labels,
    )


def _operation_partition_labels(
    original: QuantumCircuit,
    result: ScheduleResult,
) -> dict[int, Hashable]:
    """Assign each non-barrier operation to a capacity-bounded fragment label."""
    events_by_original = {ev.original_index: ev for ev in result.trace}
    initial = result.extra.get("initial_resident")
    if initial is None:
        initial = result.trace[0].resident_before
    resident = set(initial)
    epoch = 0
    labels: dict[int, Hashable] = {}

    for i, inst in enumerate(original.data):
        op_name = inst.operation.name
        qubits = [original.find_bit(q).index for q in inst.qubits]

        # Barriers are split/handled by qiskit-addon-cutting and should not
        # force additional wire-label transitions in the lowering.
        if op_name == "barrier" or not qubits:
            continue

        if len(qubits) == 2 and i in events_by_original:
            event = events_by_original[i]
            if event.loads:
                epoch += 1
            resident = set(event.resident_after)
            labels[i] = f"epoch_{epoch}"
        elif len(qubits) == 1:
            q = qubits[0]
            if q in resident:
                labels[i] = f"epoch_{epoch}"
            else:
                labels[i] = f"local_{i}_{q}"
        else:
            # The current experiments use one- and two-qubit operations plus
            # barriers.  If a future workload contains a wider operation, keep
            # it in a singleton label so validation fails only if Qiskit cannot
            # decompose it, not because labels exceeded capacity silently.
            labels[i] = f"wide_{i}"

    return labels


def _insert_label_transition_cuts(
    original: QuantumCircuit,
    op_labels: dict[int, Hashable],
) -> tuple[QuantumCircuit, list[list[Hashable]], int]:
    """Insert ``CutWire`` whenever a qubit crosses partition labels."""
    qc = QuantumCircuit(original.num_qubits)
    current_label: list[Hashable | None] = [None] * original.num_qubits
    segment_labels: list[list[Hashable]] = [[] for _ in range(original.num_qubits)]
    interface_count = 0

    for i, inst in enumerate(original.data):
        qubits = [original.find_bit(q).index for q in inst.qubits]
        label = op_labels.get(i)

        if label is not None:
            for q in qubits:
                if current_label[q] is None:
                    current_label[q] = label
                    segment_labels[q].append(label)
                elif current_label[q] != label:
                    qc.append(CutWire(), [q])
                    interface_count += 1
                    current_label[q] = label
                    segment_labels[q].append(label)

        qc.append(inst.operation, qubits)

    return qc, segment_labels, interface_count


def _expand_cut_wires_to_moves(
    annotated: QuantumCircuit,
    segment_labels: list[list[Hashable]],
) -> QuantumCircuit:
    """Expand ``CutWire`` instructions into QPD ``Move`` gates.

    qiskit-addon-cutting's public ``cut_wires`` helper allocates one new wire
    per ``CutWire`` and inserts a QPD ``Move`` between consecutive segments.
    In version 0.10.0 its internal preallocation counts assume cut instructions
    on the same input wire appear contiguously. Trace lowerings naturally
    interleave cuts across wires, so we perform the same transformation here
    using the already known per-wire segment structure.
    """
    segment_indices: list[list[int]] = []
    next_index = 0
    for labels in segment_labels:
        width = max(1, len(labels))
        indices = list(range(next_index, next_index + width))
        segment_indices.append(indices)
        next_index += width

    expanded = QuantumCircuit(next_index)
    active_segment = [0] * annotated.num_qubits

    for inst in annotated.data:
        qubits = [annotated.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "cut_wire":
            q = qubits[0]
            old_idx = segment_indices[q][active_segment[q]]
            active_segment[q] += 1
            new_idx = segment_indices[q][active_segment[q]]
            expanded.append(
                TwoQubitQPDGate.from_instruction(Move()),
                [old_idx, new_idx],
            )
        else:
            mapped_qubits = [segment_indices[q][active_segment[q]] for q in qubits]
            expanded.append(inst.operation, mapped_qubits)

    return expanded


if __name__ == "__main__":
    from circuits import make_qaoa_maxcut, make_qft

    # Quick test
    qc = make_qaoa_maxcut(24, seed=42)
    for policy in ["Random", "LRU", "LR-Gate", "Lookahead"]:
        result = schedule(qc, M=20, policy=policy)
        print(f"{policy:>10}: K={result.K}")
