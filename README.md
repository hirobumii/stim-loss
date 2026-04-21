# stim-loss

> **This is a fork of [Stim](https://github.com/quantumlib/Stim) that adds native atom-loss simulation for neutral atom quantum computing platforms.**

## What's New in This Fork

Neutral atom platforms can distinguish three qubit outcomes: **|0⟩**, **|1⟩**, and **lost** (atom absent). Standard Pauli channels cannot represent atom loss faithfully — conflating loss with |0⟩ destroys the soft-information advantage that erasure decoders rely on.

This fork introduces three new instructions:

| Instruction | Description |
|---|---|
| `LOSS_ERROR(p)` | With probability `p`, permanently lose a qubit. The loss state is attached to the qubit and therefore moves through `SWAP`. Subsequent non-`SWAP` two-qubit gates on a lost qubit are silently skipped. |
| `HERALDED_LOSS(p)` | Same as above, but appends a herald bit to the measurement record (1 = lost, 0 = alive). |
| `M_LOSS` | Read the loss state of a qubit directly (1 = lost, 0 = alive). Does not disturb the quantum state. |

### Three-Outcome Readout Pattern

Combine `M_LOSS` and `M` to implement full ternary readout. For circuits containing loss instructions, use `stim.TableauSimulator` instead of `circuit.compile_sampler()`:

```python
import stim

sim = stim.TableauSimulator()
sim.do_circuit(stim.Circuit("""
    LOSS_ERROR(0.05) 0
    M_LOSS 0   # rec[-2]: 1 = lost,  0 = alive
    M 0        # rec[-1]: qubit value (only meaningful if alive)
"""))
sample = sim.current_measurement_record()
# sample[0] == 1  →  atom lost
# sample[0] == 0, sample[1] == 0  →  alive, |0⟩
# sample[0] == 0, sample[1] == 1  →  alive, |1⟩
```

### More Examples

```python
# Heralded loss
circuit = stim.Circuit("""
    H 0
    HERALDED_LOSS(0.1) 0   # rec[0]: herald bit
    M 0                    # rec[1]: qubit value
""")

# Loss isolation: CZ does not affect the surviving qubit
circuit = stim.Circuit("""
    H 1
    LOSS_ERROR(1.0) 0   # qubit 0 definitely lost
    CZ 0 1              # silently skipped for qubit 0
    M 1                 # qubit 1 is completely unaffected
""")

# Loss follows the qubit through SWAP
sim = stim.TableauSimulator()
sim.do_circuit(stim.Circuit("""
    LOSS_ERROR(1.0) 0
    SWAP 0 1
    M_LOSS 0 1          # rec[0] = 0, rec[1] = 1
"""))

# Reset restores a lost qubit
circuit = stim.Circuit("""
    LOSS_ERROR(1.0) 0
    R 0    # clears loss state; qubit 0 is active again
    H 0
    CZ 0 1
""")

# Classical feedforward on loss
circuit = stim.Circuit("""
    LOSS_ERROR(1.0) 0
    M_LOSS 0           # rec[-1] = 1 (lost)
    CX rec[-1] 1       # conditional correction triggered by loss
""")
```

### Limitations of This Fork

- **No DEM export for loss circuits.** `LOSS_ERROR` and `HERALDED_LOSS` raise an error when `circuit.detector_error_model()` is called, because atom loss creates a dynamic entanglement graph that cannot be represented as a static Detector Error Model.
- **No compiled measurement sampling for loss circuits.** `circuit.compile_sampler()` is intentionally unsupported for circuits containing `LOSS_ERROR`, `HERALDED_LOSS`, or `M_LOSS`, because compiled sampling uses a frame-simulator path that does not model physical loss transport. Use `stim.TableauSimulator` instead.
- `M_LOSS` is safe to use in DEM export — it consumes its measurement record slot without generating any Pauli error sensitivity.
