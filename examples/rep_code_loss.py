"""
3-bit repetition code with atom loss — stim-loss demo
======================================================

Demonstrates LOSS_ERROR and M_LOSS in a 3-qubit repetition code with
neutral atom loss. The key point: M alone cannot distinguish a lost atom
from a live one. M_LOSS is essential for loss-aware decoding.

Physical setup
--------------
  Data qubits : 0, 1, 2
  Encoding    : |0_L> = |000>,  |1_L> = |111>

Why M alone is not enough
--------------------------
  After LOSS_ERROR, a lost qubit is silently isolated in |0> in the
  Pauli frame. The M gate returns the reference value (1 for our |1>
  encoding), not a reliable qubit measurement. Without M_LOSS you
  cannot tell whether the atom is present or absent.

Measurement record per shot (6 bits)
--------------------------------------
  [0,1,2]  M_LOSS q0, q1, q2   (1 = atom lost,  0 = atom alive)
  [3,4,5]  M      q0, q1, q2   (meaningful only when M_LOSS == 0)
"""

import stim
import numpy as np

# ── Parameters ─────────────────────────────────────────────────────────────────

LOSS_PROB = 0.05    # 5% per-qubit atom loss probability
N_SHOTS   = 20_000

# ── Circuit ────────────────────────────────────────────────────────────────────

circuit = stim.Circuit("""
    # Encode logical |1_L> = |111>
    X 0 1 2

    # Atom loss: each qubit independently lost with probability p
    LOSS_ERROR(0.05) 0 1 2

    # 3-state readout per qubit
    M_LOSS 0 1 2   # loss flag  : 1 = atom lost,  0 = atom alive
    M      0 1 2   # qubit value: reliable only when the loss flag is 0
""")

print("Circuit:")
print(circuit)

# ── Sampling ───────────────────────────────────────────────────────────────────

sampler = circuit.compile_sampler()
samples = sampler.sample(shots=N_SHOTS)

loss = samples[:, 0:3]   # M_LOSS results
meas = samples[:, 3:6]   # M      results

# ── Decoders ───────────────────────────────────────────────────────────────────

def decode_naive(meas_bits):
    """Majority vote on raw M values — ignores loss information."""
    return int(meas_bits.sum() >= 2)

def decode_loss_aware(loss_bits, meas_bits):
    """
    Majority vote on surviving qubits only.

    Returns:
        1 or 0  — decoded logical value
       -1       — too many losses to decode (≥2 lost)
    """
    n_lost = int(loss_bits.sum())
    if n_lost == 0:
        return int(meas_bits.sum() >= 2)
    elif n_lost == 1:
        survivors = meas_bits[loss_bits == 0]
        # Two surviving qubits must agree
        return int(survivors[0] == survivors[1] == 1)
    else:
        return -1   # undecidable

naive_decoded     = np.array([decode_naive(meas[i]) for i in range(N_SHOTS)])
aware_decoded     = np.array([decode_loss_aware(loss[i], meas[i]) for i in range(N_SHOTS)])

# ── Statistics ─────────────────────────────────────────────────────────────────

n_lost_per_shot = loss.sum(axis=1)

print(f"\n=== Loss distribution  (p_loss={LOSS_PROB}, N={N_SHOTS:,}) ===")
for k in range(4):
    n = int((n_lost_per_shot == k).sum())
    print(f"  {k} qubit(s) lost : {n:6,}  ({100*n/N_SHOTS:5.1f}%)")

# Naive decoder
n_naive_correct = int((naive_decoded == 1).sum())
n_naive_error   = int((naive_decoded == 0).sum())

# Loss-aware decoder
aware_decodable  = aware_decoded >= 0
n_aware_correct  = int((aware_decoded[aware_decodable] == 1).sum())
n_aware_error    = int((aware_decoded[aware_decodable] == 0).sum())
n_undecidable    = int((aware_decoded == -1).sum())

print(f"\n=== Decoding logical |1_L>  (correct answer = 1) ===")
print(f"\n  Naive majority vote (no M_LOSS):")
print(f"    Correct   : {n_naive_correct:6,}  ({100*n_naive_correct/N_SHOTS:5.1f}%)")
print(f"    Error     : {n_naive_error:6,}  ({100*n_naive_error/N_SHOTS:5.1f}%)")

print(f"\n  Loss-aware decoder (uses M_LOSS):")
print(f"    Correct   : {n_aware_correct:6,}  ({100*n_aware_correct/N_SHOTS:5.1f}%)")
print(f"    Error     : {n_aware_error:6,}  ({100*n_aware_error/N_SHOTS:5.1f}%)")
print(f"    Undecidable (≥2 lost): {n_undecidable:6,}  ({100*n_undecidable/N_SHOTS:5.1f}%)")

# ── Single-qubit loss detail ────────────────────────────────────────────────────

print(f"\n=== Single-qubit loss events: M and M_LOSS readout ===")
single_loss_mask = n_lost_per_shot == 1
lost_idx  = np.argmax(loss[single_loss_mask], axis=1)
meas_sl   = meas[single_loss_mask]
loss_sl   = loss[single_loss_mask]

for q in range(3):
    q_mask    = lost_idx == q
    m_vals    = meas_sl[q_mask, q]    # M value of the lost qubit
    ml_vals   = loss_sl[q_mask, q]    # M_LOSS value of the lost qubit (always 1)
    n_q       = q_mask.sum()
    # Show unique (M_LOSS, M) pairs and their counts
    pairs, counts = np.unique(
        np.stack([ml_vals, m_vals], axis=1), axis=0, return_counts=True
    )
    pair_str = ", ".join(
        f"(M_LOSS={p[0]}, M={p[1]}): {c} shots" for p, c in zip(pairs, counts)
    )
    print(f"  q{q} lost ({n_q} shots): {pair_str}")

print()
print("Note: M alone returns the reference value (1 for |1_L> encoding),")
print("making loss indistinguishable from a live |1> qubit without M_LOSS.")
