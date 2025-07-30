import os
import time
import numpy as np
import scipy.sparse as sp
import joblib
import psutil
from variational_lse_solver.var_lse_solver import VarLSESolver


def generate_diag_dominant_sparse_matrix(dim, sparsity=0.01):
    print("    [??] ????????...")
    A = sp.random(dim, dim, density=sparsity, format='csr', data_rvs=np.random.rand)
    abs_row_sums = np.abs(A).sum(axis=1).A1
    A.setdiag(abs_row_sums + 1.0)
    return A


def run_sample(size, idx, local, threshold, steps, epochs, queue=None):
    try:
        dim = 2 ** size
        print(f"  [??] >> ?? {idx}: dim = {dim}, steps = {steps}, epochs = {epochs}")

        mem = psutil.virtual_memory()
        print(f"  [??] ????: {mem.available / 1024**2:.1f} MB")
        if mem.available < 1.5 * dim * dim * 8:
            raise RuntimeError("????,??????")

        print("  [??] ?????? A...")
        A_sparse = generate_diag_dominant_sparse_matrix(dim, sparsity=0.01)

        print("  [??] ???????...")
        A_dense = A_sparse.toarray()

        print("  [??] ???? b...")
        b = np.ones(dim) / np.sqrt(dim)

        print("  [??] ??? VarLSESolver...")
        lse = VarLSESolver(
            A_dense, b, coeffs=None, method="direct",
            local=local, steps=steps, epochs=epochs, threshold=threshold,
        )

        print("  [??] ????...")
        solution, all_weights, all_losses = lse.solve(
            save_epoch_dir=f"epoch_data_size{size}_sample", sample_idx=idx
        )
        print("  [??] ?????")

        # ????
        try:
            os.makedirs(f"data/sparse_{size}", exist_ok=True)
            prefix = f"data/sparse_{size}/sparse{size}_{idx}"

            joblib.dump(A_sparse, f"{prefix}_matrix.pkl")
            joblib.dump(b, f"{prefix}_b.pkl")
            joblib.dump(solution, f"{prefix}_w.pkl")
            joblib.dump(all_weights, f"{prefix}_epoch_weights.pkl")
            joblib.dump(all_losses, f"{prefix}_epoch_losses.pkl")

            print(f"  [??] ?? {idx} ?????")
        except Exception as e:
            print(f"  [??] ?? {idx} ????: {e}")

        if queue:
            queue.put((idx, True, None))

    except Exception as e:
        print(f"  [??] ?? {idx} ????: {e}")
        if queue:
            queue.put((idx, False, str(e)))


if __name__ == "__main__":
    size = 8  # ???:?? 2^10 = 1024 ?????
    start_idx = 600
    total_new_samples = 600
    end_idx = start_idx + total_new_samples

    steps = 10
    epochs = 10
    local = False
    threshold = 1e-3

    print(f"\n==== ???? sparse{size},???? {start_idx} ? {end_idx - 1} ====")
    start_time = time.time()

    for i in range(start_idx, end_idx):
        print(f"\n[>>] --> ?????? {i}/{end_idx - 1}")
        try:
            run_sample(size, i, local, threshold, steps, epochs)
        except Exception as e:
            print(f"  [??] ?? {i} ??????: {e}")

    elapsed = time.time() - start_time
    print(f"\n??  ????????:? {start_idx} ? {end_idx - 1},?? {elapsed:.2f} ?")
