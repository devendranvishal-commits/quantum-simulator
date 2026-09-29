"""
BUILD-YOUR-OWN QUANTUM COMPUTER (in software)
===============================================
Built for Vishal, OSU Tech Innovation / BuildX prep project.

Why this file exists:
    We couldn't install Qiskit (Anthropic's sandbox network policy blocks
    pypi.org for this session) so instead of faking it, we built the real
    thing ourselves using nothing but numpy. This is not a toy - it is a
    genuine statevector quantum simulator, the same technique real quantum
    researchers use when they don't have hardware time. Then we run a real
    quantum algorithm (QAOA) on it to solve a business problem: splitting
    delivery stops between two trucks to balance the workload - the same
    "routing/optimization" idea from our brainstorm, in miniature.

Three parts:
    PART 1: A quantum simulator from scratch (qubits, gates, measurement)
    PART 2: Proof it's really quantum (superposition + entanglement demos)
    PART 3: QAOA - a real hybrid quantum-classical algorithm - solving a
            tiny "balance the delivery routes between 2 trucks" problem,
            checked against the brute-force correct answer.

Run it with:  python3 quantum_demo.py
"""

import numpy as np
from itertools import product
from scipy.optimize import minimize

np.set_printoptions(precision=3, suppress=True)

# ======================================================================
# PART 1: THE QUANTUM SIMULATOR
# ======================================================================
# A quantum computer with n qubits is described by a single vector of
# 2^n complex numbers (the "statevector"). Each of the 2^n numbers is the
# amplitude of one possible outcome (00...0, 00...1, ..., 11...1).
# The probability of measuring a given outcome is |amplitude|^2.

# The basic single-qubit gates, written as 2x2 matrices.
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)              # bit flip
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)  # superposition


def RX(theta):
    """Rotation around the X axis of the qubit - used as the QAOA 'mixer'."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def RZ(theta):
    """Rotation around the Z axis - used to encode the 'cost' of a solution."""
    return np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]], dtype=complex)


def apply_single_qubit_gate(state, gate, qubit, n):
    """
    Apply a 2x2 gate to one qubit out of n, by building the full 2^n x 2^n
    matrix as a tensor (Kronecker) product: I ⊗ I ⊗ ... ⊗ gate ⊗ ... ⊗ I
    """
    op = np.array([[1]], dtype=complex)
    for q in range(n):
        op = np.kron(op, gate if q == qubit else I2)
    return op @ state


def apply_cnot(state, control, target, n):
    """
    CNOT: if the control qubit is |1>, flip the target qubit.
    This is the gate that creates ENTANGLEMENT - the spooky link between
    qubits that makes them behave as one system instead of independent coins.
    """
    dim = 2 ** n
    new_state = np.zeros(dim, dtype=complex)
    for i in range(dim):
        bits = list(format(i, f"0{n}b"))
        if bits[control] == "1":
            bits[target] = "1" if bits[target] == "0" else "0"
        j = int("".join(bits), 2)
        new_state[j] += state[i]
    return new_state


def zero_state(n):
    """Start every qubit in the |0> state (like a coin resting on heads)."""
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    return state


def measure(state, n, shots=2000, seed=None):
    """
    Simulate 'shots' repeated measurements. Each shot collapses the
    statevector into one classical outcome, with probability |amplitude|^2.
    This is the only step where quantum randomness enters classical reality.
    """
    rng = np.random.default_rng(seed)
    probs = np.abs(state) ** 2
    probs = probs / probs.sum()  # guard against float rounding
    outcomes = rng.choice(len(state), size=shots, p=probs)
    counts = {}
    for o in outcomes:
        key = format(o, f"0{n}b")
        counts[key] = counts.get(key, 0) + 1
    return counts


# ======================================================================
# PART 2: PROVE IT'S ACTUALLY QUANTUM
# ======================================================================

def demo_superposition():
    print("=" * 70)
    print("DEMO 1: SUPERPOSITION")
    print("=" * 70)
    print("A classical bit is 0 or 1. A qubit put through a Hadamard gate (H)")
    print("becomes BOTH at once, and only 'picks' a value when measured.\n")

    n = 1
    state = zero_state(n)
    state = apply_single_qubit_gate(state, H, qubit=0, n=n)
    print(f"Statevector after H: {state}")
    print("(Equal amplitude on |0> and |1> -> 50/50 chance on measurement)\n")

    counts = measure(state, n, shots=2000, seed=42)
    print(f"2000 simulated measurements: {counts}")
    print("-> roughly 50/50, exactly as quantum mechanics predicts.\n")


def demo_entanglement():
    print("=" * 70)
    print("DEMO 2: ENTANGLEMENT (the Bell state)")
    print("=" * 70)
    print("H on qubit 0, then CNOT(0 -> 1). This links two qubits so they")
    print("always agree when measured, even though each one alone is random.\n")

    n = 2
    state = zero_state(n)
    state = apply_single_qubit_gate(state, H, qubit=0, n=n)
    state = apply_cnot(state, control=0, target=1, n=n)
    print(f"Statevector: {state}")
    print("(Only |00> and |11> have amplitude - |01> and |10> are impossible)\n")

    counts = measure(state, n, shots=2000, seed=7)
    print(f"2000 simulated measurements: {counts}")
    print("-> only ever 00 or 11, ~50/50 - the two qubits are perfectly")
    print("   correlated despite each one individually being a coin flip.\n")


# ======================================================================
# PART 3: QAOA - SOLVING A REAL BUSINESS PROBLEM
# ======================================================================
# The problem: you have delivery stops connected in a network (edges =
# routes between stops with some "cost" - could be distance, time, or
# traffic). You have TWO trucks and want to split the stops between them
# so that the total cost of routes that CROSS between the two trucks'
# territories is MAXIMIZED (this is the classic "Max-Cut" problem, the
# standard first real-world target for QAOA). In practice this same math
# is used for balancing workloads, partitioning networks, and — with a
# different cost function — the vehicle routing problems we talked about
# for a small delivery/logistics business.
#
# We represent "which truck each stop is assigned to" as one qubit per
# stop: |0> = Truck A, |1> = Truck B.

# A tiny delivery network: 4 stops, edges = routes with a "cost" weight.
EDGES = [
    (0, 1, 1.0),
    (1, 2, 1.0),
    (2, 3, 1.0),
    (3, 0, 1.0),
    (0, 2, 0.5),
]
N_STOPS = 4


def cutcost_classical(bitstring, edges):
    """The actual value we're trying to maximize: cost of edges that CROSS
    between the two truck groups (i.e., the two endpoints are assigned to
    different trucks)."""
    total = 0.0
    for (i, j, w) in edges:
        if bitstring[i] != bitstring[j]:
            total += w
    return total


def brute_force_optimum(n, edges):
    """Check every possible assignment - fine for tiny n, this is our
    'ground truth' to grade the quantum algorithm against."""
    best_val, best_bits = -1, None
    for bits in product("01", repeat=n):
        val = cutcost_classical(bits, edges)
        if val > best_val:
            best_val, best_bits = val, bits
    return best_val, "".join(best_bits)


def qaoa_circuit(gammas, betas, n, edges):
    """
    Build and run one QAOA circuit:
      1. Put every qubit into superposition (H on all).
      2. Repeat p layers of:
           a. 'Cost' layer: RZ rotations that encode the edges (a ZZ-style
              interaction approximated here via correlated RZ phases).
           b. 'Mixer' layer: RX rotations that let the state explore
              different assignments.
      3. Return the final statevector.

    Note: a full ZZ interaction needs a 2-qubit entangling gate; for
    clarity here we approximate the standard QAOA cost unitary using
    CNOT-RZ-CNOT, which is the textbook decomposition of exp(-i*gamma*Z_i*Z_j).
    """
    state = zero_state(n)
    for q in range(n):
        state = apply_single_qubit_gate(state, H, qubit=q, n=n)

    for gamma, beta in zip(gammas, betas):
        # Cost layer: for every edge, apply exp(-i * gamma * w * Z_i Z_j)
        for (i, j, w) in edges:
            state = apply_cnot(state, i, j, n)
            state = apply_single_qubit_gate(state, RZ(2 * gamma * w), j, n)
            state = apply_cnot(state, i, j, n)
        # Mixer layer: explore other assignments
        for q in range(n):
            state = apply_single_qubit_gate(state, RX(2 * beta), q, n)

    return state


def expected_cost(params, n, edges, p):
    """The number the classical optimizer is trying to MAXIMIZE (we minimize
    its negative). This is the 'hybrid' part of QAOA: a classical optimizer
    (scipy) steers the quantum circuit's parameters."""
    gammas, betas = params[:p], params[p:]
    state = qaoa_circuit(gammas, betas, n, edges)
    probs = np.abs(state) ** 2
    total = 0.0
    for idx, prob in enumerate(probs):
        bits = format(idx, f"0{n}b")
        total += prob * cutcost_classical(bits, edges)
    return -total  # minimize negative = maximize cost


def run_qaoa(n, edges, p=2, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi, size=2 * p)
    result = minimize(expected_cost, x0, args=(n, edges, p), method="COBYLA",
                       options={"maxiter": 300})
    gammas, betas = result.x[:p], result.x[p:]
    final_state = qaoa_circuit(gammas, betas, n, edges)
    counts = measure(final_state, n, shots=4000, seed=seed)
    best_bits = max(counts, key=counts.get)
    return best_bits, counts, -result.fun


def demo_qaoa_routing():
    print("=" * 70)
    print("DEMO 3: QAOA solving a delivery-routing-style problem")
    print("=" * 70)
    print(f"{N_STOPS} delivery stops, routes between them with a 'cost':")
    for (i, j, w) in EDGES:
        print(f"   stop {i} <-> stop {j}   cost={w}")
    print("\nGoal: split stops between Truck A (0) and Truck B (1) to")
    print("MAXIMIZE the total cost of routes that cross between the trucks")
    print("(the standard Max-Cut problem - the textbook first QAOA use case,")
    print("and the same style of math used in real vehicle-routing tools).\n")

    best_val, best_bits = brute_force_optimum(N_STOPS, EDGES)
    print(f"Brute-force ground truth: assignment {best_bits} -> value {best_val}\n")

    print("Running QAOA (quantum circuit + classical optimizer loop)...")
    q_bits, counts, q_val = run_qaoa(N_STOPS, EDGES, p=2, seed=1)
    print(f"QAOA's most-measured assignment: {q_bits} -> value "
          f"{cutcost_classical(q_bits, EDGES)}")
    print(f"QAOA's expected value from its optimized circuit: {q_val:.3f}")
    print(f"\nMeasurement distribution over 4000 shots (top 5):")
    for bits, c in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
        print(f"   {bits}: {c} times ({100*c/4000:.1f}%)")

    match = "MATCHES the optimal split!" if cutcost_classical(q_bits, EDGES) == best_val else \
            "close, but not the exact optimum on this run (normal for small QAOA depth)"
    print(f"\nResult: QAOA's top answer {match}")


if __name__ == "__main__":
    demo_superposition()
    demo_entanglement()
    demo_qaoa_routing()
