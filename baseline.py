import os
import numpy as np
import joblib
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from variational_lse_solver.var_lse_solver import VarLSESolver

# 配置路径
base_dir = "C:\Users\yangy\OneDrive\Desktop\pvls\data\synthetic matrices"
test_sets = {
    "test_8": {
        "matrix": f"{base_dir}/sparse8_matrix_list.pkl",
        "answer": f"{base_dir}/sparse8_b_list.pkl",
    },
    "test_9": {
        "matrix": f"{base_dir}/sparse9_matrix_list.pkl",
        "answer": f"{base_dir}/sparse9_b_list.pkl",
    },
    "test_10": {
        "matrix": f"{base_dir}/sparse10_matrix_list.pkl",
        "answer": f"{base_dir}/sparse10_b_list.pkl",
    },
}

output_base_dir = "./vls_baseline"
os.makedirs(output_base_dir, exist_ok=True)

# 方法一：PCA初始化
def pca_init(matrix, qn):
    pca = PCA(n_components=qn * 3)
    try:
        pca.fit(matrix)
        init = pca.components_[0]
    except:
        init = np.random.rand(qn * 3)
    return init_to_weight(init, qn)

# 方法二：最小范数初始化
def min_norm_init(matrix, b, qn):
    try:
        x = np.linalg.pinv(matrix) @ b
    except:
        x = np.random.rand(matrix.shape[1])
    return init_to_weight(x, qn)

# 方法三：均值初始化
def uniform_mean_init(matrix, qn):
    mean_row = np.mean(matrix, axis=1)
    return init_to_weight(mean_row, qn)

# 映射为(1, qn, 3)的weight
def init_to_weight(init, qn):
    flat = np.copy(init).flatten()
    if len(flat) < qn * 3:
        flat = np.pad(flat, (0, qn * 3 - len(flat)))
    else:
        flat = flat[:qn * 3]
    flat = np.mod(flat, 2 * np.pi)
    return flat.reshape((1, qn, 3))

# 计算loss
def compute_loss(matrix, answer, weight, steps=100, threshold=1e-5):
    lse = VarLSESolver(a=matrix, b=answer, method="direct", local=False, epochs=1, threshold=threshold)
    initial_loss = lse.test_prediction_loss(weight).item()
    training_losses = lse.continue_training_from_predicted(weight, N=steps)
    return initial_loss, training_losses

# 可选方法集合
methods = {
    "pca": pca_init,
    "minnorm": min_norm_init,
    "uniform": uniform_mean_init,
}

# 遍历数据集
for name, paths in test_sets.items():
    print(f"\n🔎 Processing {name}")
    matrix_list = joblib.load(paths["matrix"])
    answer_list = joblib.load(paths["answer"])
    matrix_list = [m.toarray() if hasattr(m, "toarray") else m for m in matrix_list]

    for method_name, method_fn in methods.items():
        print(f"   └── [{method_name}] evaluating...")
        init_losses, train_losses = [], []

        for idx, (matrix, answer) in enumerate(tqdm(zip(matrix_list, answer_list), total=len(answer_list))):
            qn = int(np.log2(answer.shape[0]))
            if method_name == "minnorm":
                weight = method_fn(matrix, answer, qn)
            else:
                weight = method_fn(matrix, qn)
            init_loss, t_losses = compute_loss(matrix, answer, weight)
            init_losses.append(init_loss)
            train_losses.append(t_losses)

        out_dir = os.path.join(output_base_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(init_losses, os.path.join(out_dir, f"{method_name}_initial_losses.pkl"))
        joblib.dump(train_losses, os.path.join(out_dir, f"{method_name}_training_losses.pkl"))
