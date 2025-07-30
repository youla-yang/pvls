import os
import joblib
import numpy as np
import torch
from variational_lse_solver.var_lse_solver import VarLSESolver



base_dir = "/home/jiang60/1/variational_lse_solver/3"
test_sets = {
    "test_8": {
        "matrix": f"{base_dir}/sparse8_matrix_list.pkl",
        "answer": f"{base_dir}/sparse8_b_list.pkl",
        "weights": f"{base_dir}/test_8_pred.pt"
    },
    "test_9": {
        "matrix": f"{base_dir}/sparse9_matrix_list.pkl",
        "answer": f"{base_dir}/sparse9_b_list.pkl",
        "weights": f"{base_dir}/test_9_pred.pt"
    },
    "test_10": {
        "matrix": f"{base_dir}/sparse10_matrix_list.pkl",
        "answer": f"{base_dir}/sparse10_b_list.pkl",
        "weights": f"{base_dir}/test_10_pred.pt"
    }
}

output_base_dir = "./vls"
os.makedirs(output_base_dir, exist_ok=True)

def compute_random_loss(matrix, answer, threshold):
    lse = VarLSESolver(a=matrix, b=answer, method="direct", local=False, epochs=1, threshold=threshold)
    loss, _ = lse.test_random_initialized_loss()
    return loss.item()

def compute_predicted_loss(matrix, answer, weight, threshold):
    lse = VarLSESolver(a=matrix, b=answer, method="direct", local=False, epochs=1, threshold=threshold)
    return lse.test_prediction_loss(weight).item()

def continue_training(matrix, answer, weight, steps=100, threshold=1e-5):
    lse = VarLSESolver(a=matrix, b=answer, method="direct", local=False, epochs=1, threshold=threshold)
    return lse.continue_training_from_predicted(weight, N=steps)

for name, paths in test_sets.items():
    print(f"\n🔄 正在处理 {name}")
    matrix_list = joblib.load(paths["matrix"])
    answer_list = joblib.load(paths["answer"])
    weight_tensor = torch.load(paths["weights"], weights_only=False)  # shape [N, qn, 3]

    # 稀疏矩阵转 dense
    matrix_list = [m.toarray() if hasattr(m, "toarray") else m for m in matrix_list]

    random_losses = []
    predicted_losses = []
    continued_training_losses = []
    random_initial_training_losses = []

    for idx, (matrix, answer) in enumerate(zip(matrix_list, answer_list)):
        qn = int(np.log2(answer.shape[0]))

        # random init
        rand_weight = np.random.uniform(0, 2 * np.pi, size=(1, qn, 3))
        random_loss = compute_random_loss(matrix, answer, threshold=1e-5)
        random_losses.append(random_loss)
        rand_train_loss = continue_training(matrix, answer, rand_weight, steps=100)
        random_initial_training_losses.append(rand_train_loss)

        if idx >= weight_tensor.shape[0]:
            print(f"⚠️ 超出预测权重范围，跳过 index {idx}")
            continue

        # predicted weight 自动补维
        pred_weight = weight_tensor[idx]
        if pred_weight.shape[0] < qn:
            pad = torch.zeros((qn - pred_weight.shape[0], 3))
            pred_weight = torch.cat([pred_weight, pad], dim=0)
        pred_weight = pred_weight[:qn].unsqueeze(0)

        predicted_loss = compute_predicted_loss(matrix, answer, pred_weight, threshold=1e-5)
        predicted_losses.append(predicted_loss)
        continued_loss = continue_training(matrix, answer, pred_weight, steps=100)
        continued_training_losses.append(continued_loss)

        print(f"[{idx}] random: {random_loss:.6f}, predicted: {predicted_loss:.6f}")

    # 保存结果
    output_dir = os.path.join(output_base_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(random_losses, os.path.join(output_dir, "random_initial_losses.pkl"))
    joblib.dump(predicted_losses, os.path.join(output_dir, "predicted_initial_losses.pkl"))
    joblib.dump(continued_training_losses, os.path.join(output_dir, "continued_training_losses.pkl"))
    joblib.dump(random_initial_training_losses, os.path.join(output_dir, "random_initial_training_losses.pkl"))

    print(f"✅ {name} 保存完毕。")

