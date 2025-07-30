import os
import time
import numpy as np
import joblib
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from variational_lse_solver.var_lse_solver import VarLSESolver

def load_sparse_matrices_from_dir(directory):
    pkl_files = sorted(
        [f for f in os.listdir(directory) if f.endswith("_matrix.pkl")],
        key=lambda x: x.lower()
    )
    matrices = [joblib.load(os.path.join(directory, f)) for f in pkl_files]
    return matrices

def test_for_loaded_matrix(A_sparse, local: bool = False, threshold: float = 1e-5):
    A_dense = A_sparse.toarray()
    dim = A_dense.shape[0]
    b = np.ones(dim) / np.sqrt(dim)

    lse = VarLSESolver(
        A_dense,
        b,
        coeffs=None,
        method="direct",
        local=local,
        steps=100,
        epochs=5,
        threshold=threshold,
    )
    solution, weight = lse.solve()  # ? ????
    return A_sparse, b, weight

def generate_one_sample_with_progress(idx, total, A_sparse, local=False, threshold=1e-5):
    if idx % 10 == 0:
        print(f"    > Processing sample {idx + 1}/{total}")
    return test_for_loaded_matrix(A_sparse, local, threshold)

if __name__ == "__main__":
    # 修改此处：pkl 稀疏矩阵路径
    matrix_dir = "C:\Users\yangy\OneDrive\Desktop\pvls\data\synthetic matrices"  # 或 "/mnt/data" 或绝对路径
    output_prefix = "sparse_pkl_data"
    n_jobs = 1
    threshold = 1e-5

    matrices = load_sparse_matrices_from_dir(matrix_dir)
    sample_count = len(matrices)

    print(f"\n==== Generating {sample_count} samples from .pkl matrices ====")
    start = time.time()

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(generate_one_sample_with_progress)(
            i, sample_count, matrices[i], False, threshold
        )
        for i in range(sample_count)
    )

    matrix_list, b_list, w_list = zip(*results)

    joblib.dump(matrix_list, f"{output_prefix}_matrix.pkl")
    joblib.dump(b_list, f"{output_prefix}_b.pkl")
    joblib.dump(w_list, f"{output_prefix}_w.pkl")

    elapsed = time.time() - start
    print(f"✅ Finished in {elapsed:.2f}s - saved matrix/b/w to {output_prefix}_*.pkl")

