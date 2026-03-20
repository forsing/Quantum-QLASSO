"""
QLASSO - Quantum Lasso Regression
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/Users/4c/Desktop/GHQ/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
ALPHA_L1 = 0.001
MAX_COORD_ITER = 200


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def compute_quantum_kernel():
    n_states = 1 << NUM_QUBITS
    fmap = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)

    statevectors = []
    for v in range(n_states):
        feat = value_to_features(v)
        circ = fmap.assign_parameters(feat)
        sv = Statevector.from_instruction(circ)
        statevectors.append(sv)

    K = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i, n_states):
            fid = abs(statevectors[i].inner(statevectors[j])) ** 2
            K[i, j] = fid
            K[j, i] = fid

    return K


def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def quantum_lasso(K, y, alpha=ALPHA_L1, max_iter=MAX_COORD_ITER):
    n = K.shape[0]
    w = np.zeros(n)

    for iteration in range(max_iter):
        for j in range(n):
            r_j = y - K @ w + K[:, j] * w[j]
            z_j = np.dot(K[:, j], r_j) / (np.dot(K[:, j], K[:, j]) + 1e-10)
            w[j] = soft_threshold(z_j, alpha)

    pred = K @ w
    return pred, w


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- Kvantni kernel (ZZFeatureMap, {NUM_QUBITS}q, reps=1) ---")
    K = compute_quantum_kernel()
    print(f"  Kernel matrica: {K.shape}, rang: {np.linalg.matrix_rank(K)}")

    print(f"\n--- QLASSO po pozicijama (alpha={ALPHA_L1}, "
          f"{MAX_COORD_ITER} iter) ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        pred, w = quantum_lasso(K, y)

        n_nonzero = np.sum(np.abs(w) > 1e-8)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: "
              f"nenula={n_nonzero}/32  {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QLASSO, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()



"""
Ucitano izvucenih kombinacija: 4580
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Kvantni kernel (ZZFeatureMap, 5q, reps=1) ---
  Kernel matrica: (32, 32), rang: 32

--- QLASSO po pozicijama (alpha=0.001, 200 iter) ---
  Poz 1 [1-33]: nenula=29/32  1:0.161 | 2:0.140 | 3:0.124
  Poz 2 [2-34]: nenula=28/32  8:0.087 | 5:0.077 | 9:0.076
  Poz 3 [3-35]: nenula=31/32  13:0.065 | 12:0.064 | 14:0.062
  Poz 4 [4-36]: nenula=29/32  23:0.065 | 21:0.065 | 18:0.064
  Poz 5 [5-37]: nenula=31/32  29:0.066 | 26:0.064 | 27:0.063
  Poz 6 [6-38]: nenula=31/32  33:0.085 | 32:0.083 | 35:0.081
  Poz 7 [7-39]: nenula=30/32  7:0.185 | 38:0.154 | 37:0.134

==================================================
Predikcija (QLASSO, deterministicki, seed=39):
[1, 8, 13, 23, 29, 33, 38]
==================================================

==================================================
Predikcija (QLASSO, deterministicki, seed=39):
[1, 8, 13, 23, 29, 33, 38]
==================================================
"""



"""
QLASSO - Quantum Lasso Regression

Quantum Lasso Regression: regresija sa L1 regularizacijom
QLASSO je kvantni algoritam za selekciju varijabli.
QLASSO se sastoji od 5 qubita i 1 sloja Ry+CX+Rz rotacija.

Isti kvantni kernel (ZZFeatureMap, fidelity, 5 qubita)
Lasso (L1) regresija u kvantnom kernel prostoru: dodaje L1 penalizaciju koja forsira retke (sparse) koeficijente
Coordinate descent: 200 iteracija sa soft-thresholding operatorom po svakom koeficijentu
Pokazuje koliko nenula koeficijenata svaka pozicija koristi - sparse model identifikuje kljucne vrednosti
Za razliku od Ridge (L2) koji koristi sve vrednosti, Lasso bira samo najvaznije
Deterministicki, brz, bez treniranja kola.
"""
