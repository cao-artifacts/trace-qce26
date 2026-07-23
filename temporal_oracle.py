"""Lightweight gate-stream scheduler and exact oracle for temporal experiments."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random

from qiskit.circuit import QuantumCircuit


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: Tuple[int, ...]

    @property
    def arity(self) -> int:
        return len(self.qubits)


class ReplacementPolicy:
    def reset(self, gates: Sequence[Gate], capacity: int) -> None:
        self.gates = list(gates)
        self.capacity = capacity

    def on_execute_two_qubit_gate(self, gate_index: int, gate: Gate) -> None:
        return

    def select_victim(self, resident: Tuple[int, ...], gate_index: int, protected: set[int]) -> int:
        raise NotImplementedError


class RandomPolicy(ReplacementPolicy):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select_victim(self, resident: Tuple[int, ...], gate_index: int, protected: set[int]) -> int:
        candidates = [q for q in resident if q not in protected]
        return self._rng.choice(sorted(candidates))


class LRUPolicy(ReplacementPolicy):
    def reset(self, gates: Sequence[Gate], capacity: int) -> None:
        super().reset(gates, capacity)
        self.last_use: Dict[int, int] = defaultdict(lambda: -1)

    def on_execute_two_qubit_gate(self, gate_index: int, gate: Gate) -> None:
        for q in gate.qubits:
            self.last_use[q] = gate_index

    def select_victim(self, resident: Tuple[int, ...], gate_index: int, protected: set[int]) -> int:
        candidates = sorted(q for q in resident if q not in protected)
        return min(candidates, key=lambda q: (self.last_use[q], q))


class _FutureUseQueues(ReplacementPolicy):
    def reset(self, gates: Sequence[Gate], capacity: int) -> None:
        super().reset(gates, capacity)
        self.future: Dict[int, Deque[int]] = defaultdict(deque)
        for i, gate in enumerate(gates):
            if gate.arity == 2:
                for q in gate.qubits:
                    self.future[q].append(i)

    def on_execute_two_qubit_gate(self, gate_index: int, gate: Gate) -> None:
        for q in gate.qubits:
            if self.future[q] and self.future[q][0] == gate_index:
                self.future[q].popleft()


class LRGatePolicy(_FutureUseQueues):
    def __init__(self, window: int) -> None:
        self.window = window

    def select_victim(self, resident: Tuple[int, ...], gate_index: int, protected: set[int]) -> int:
        candidates = sorted(q for q in resident if q not in protected)
        horizon = gate_index + self.window

        def score(q: int) -> float:
            queue = self.future[q]
            if not queue:
                return math.inf
            nxt = queue[0]
            return nxt if nxt <= horizon else math.inf

        return max(candidates, key=lambda q: (score(q), -q))


class LookaheadPolicy(_FutureUseQueues):
    def select_victim(self, resident: Tuple[int, ...], gate_index: int, protected: set[int]) -> int:
        candidates = sorted(q for q in resident if q not in protected)

        def score(q: int) -> float:
            queue = self.future[q]
            return queue[0] if queue else math.inf

        return max(candidates, key=lambda q: (score(q), -q))


def qiskit_to_gate_stream(qc: QuantumCircuit) -> List[Gate]:
    stream: List[Gate] = []
    for inst in qc.data:
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        stream.append(Gate(inst.operation.name, qubits))
    return stream


def first_m_distinct_qubits(gates: Sequence[Gate], capacity: int) -> Tuple[int, ...]:
    seen: List[int] = []
    for gate in gates:
        for q in gate.qubits:
            if q not in seen:
                seen.append(q)
                if len(seen) == capacity:
                    return tuple(seen)
    return tuple(seen)


def schedule_gate_stream(gates: Sequence[Gate], capacity: int, policy: ReplacementPolicy,
                         initial_resident: Optional[Tuple[int, ...]] = None) -> int:
    gates = list(gates)
    if initial_resident is None:
        initial_resident = first_m_distinct_qubits(gates, capacity)
    resident = list(initial_resident)
    policy.reset(gates, capacity)
    cut_count = 0

    for gate_index, gate in enumerate(gates):
        if gate.arity != 2:
            continue
        a, b = gate.qubits
        missing = [q for q in (a, b) if q not in resident]
        for q in missing:
            victim = policy.select_victim(tuple(resident), gate_index, {a, b})
            resident.remove(victim)
            resident.append(q)
            cut_count += 1
        policy.on_execute_two_qubit_gate(gate_index, gate)
    return cut_count


def extract_two_qubit_stream(gates: Sequence[Gate]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for gate in gates:
        if gate.arity == 2:
            a, b = gate.qubits
            out.append((a, b))
    return out


def exact_optimum_cut_count(two_qubit_stream: Sequence[Tuple[int, int]], capacity: int,
                            initial_resident: Optional[Tuple[int, ...]] = None) -> int:
    stream = list(two_qubit_stream)
    if initial_resident is None:
        initial_resident = first_m_distinct_qubits([Gate('g', pair) for pair in stream], capacity)
    initial_resident = tuple(sorted(initial_resident))

    @lru_cache(maxsize=None)
    def dp(i: int, resident: Tuple[int, ...]) -> int:
        if i == len(stream):
            return 0
        a, b = stream[i]
        resident_set = set(resident)
        required = {a, b}
        missing = tuple(sorted(required - resident_set))
        if not missing:
            return dp(i + 1, tuple(sorted(resident)))

        candidates = [q for q in resident if q not in required]
        best = math.inf
        for victims in combinations(candidates, len(missing)):
            new_resident = list(resident)
            for victim, incoming in zip(victims, missing):
                new_resident.remove(victim)
                new_resident.append(incoming)
            best = min(best, len(missing) + dp(i + 1, tuple(sorted(new_resident))))
        return best

    return dp(0, initial_resident)


def mean_forward_reuse_distance(two_qubit_stream: Sequence[Tuple[int, int]]) -> float:
    positions: Dict[int, List[int]] = defaultdict(list)
    for idx, (a, b) in enumerate(two_qubit_stream):
        positions[a].append(idx)
        positions[b].append(idx)
    dists: List[int] = []
    for q, pos in positions.items():
        for i in range(len(pos) - 1):
            dists.append(pos[i + 1] - pos[i])
    return float(sum(dists) / len(dists)) if dists else math.inf


def working_set_turnover(two_qubit_stream: Sequence[Tuple[int, int]], window: int = 4) -> float:
    if len(two_qubit_stream) <= window:
        return 0.0
    sets = []
    for start in range(0, len(two_qubit_stream) - window + 1):
        active = set()
        for a, b in two_qubit_stream[start:start + window]:
            active.add(a)
            active.add(b)
        sets.append(active)
    turnovers = []
    for left, right in zip(sets, sets[1:]):
        union = left | right
        if not union:
            continue
        turnovers.append(len(left ^ right) / len(union))
    return float(sum(turnovers) / len(turnovers)) if turnovers else 0.0
