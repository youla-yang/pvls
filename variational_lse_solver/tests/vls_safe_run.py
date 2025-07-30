import os
import joblib
import numpy as np
import torch
from variational_lse_solver.var_lse_solver import VarLSESolver

# ========== ???? ==========
base_dir = "/home/jiang60/1/variational_lse_solver/3"
matrix_path = os.path.join(base_dir, "sparse_pkl_data_matrix.pkl")
b_path = os.path.join(base_dir, "sparse_pkl_data_b.pkl")
pred_path = os.path.join(base_dir, "pred_list.pkl")  # joblib ????????
output_dir = os.path.join(base_dir, "vls_eval")
os.makedirs(output_dir, exist_ok=True)

# ========== ???? ==========
matrices = joblib.load(matrix_path)
b_list = joblib.load(b_path)
pred_list = joblib.load(pred_path)  # list[Tensor]

# ???????
matrices = [m.toarray() if hasattr(m, "toarray") else m for m in matrices]

# ========== ?????? ==========
def compute_random_loss(matrix, answer, threshold):
    lse = VarLSESolver(a=matrix, b=answer, method="direct", local=False, epochs=1, threshold=threshold)
    loss, _ = lse.test_random_initialized_loss()
    return float(loss)

def compute_predicted_loss(matrix, answer, weight, threshold):
    lse = VarLSESolver(a=matrix, b=answer, method="direct", local=False, epochs=1, threshold=threshold)
    return float(lse.test_prediction_loss(weight))

def continue_training(matrix, answer, weight, steps=100, threshold=1e-5):
    lse = VarLSESolver(a=matrix, b=answer, method="direct", local=False, epochs=1, threshold=threshold)
    losses = lse.continue_training_from_predicted(weight, N=steps)
    return losses  # <-- ????loss??(list[float])

# ========== ???????????loss?? ========== 
random_losses = []
predicted_losses = []
continued_loss_traces = []  # <-- ???,????????
random_init_loss_traces = []

# ========== ??? ========== 
for idx, (matrix, b) in enumerate(zip(matrices, b_list)):
    qn = int(np.log2(len(b)))
    rand_weight = np.random.uniform(0, 2 * np.pi, size=(1, qn, 3))

    rand_loss = compute_random_loss(matrix, b, threshold=1e-5)
    random_losses.append(rand_loss)

    rand_train_losses = continue_training(matrix, b, rand_weight, steps=100)
    random_init_loss_traces.append(rand_train_losses)

    if idx >= len(pred_list):
        print(f"[{idx}] ??????,??")
        continue

    pred_weight = pred_list[idx]
    if isinstance(pred_weight, torch.Tensor):
        pred_weight = pred_weight.detach().clone()

    if pred_weight.shape[0] < qn:
        pad = torch.zeros((qn - pred_weight.shape[0], 3))
        pred_weight = torch.cat([pred_weight, pad], dim=0)
    pred_weight = pred_weight[:qn].unsqueeze(0)

    pred_loss = compute_predicted_loss(matrix, b, pred_weight, threshold=1e-5)
    predicted_losses.append(pred_loss)

    pred_train_losses = continue_training(matrix, b, pred_weight, steps=100)
    continued_loss_traces.append(pred_train_losses)

    try:
        print(f"[{idx}] random: {rand_loss:.6f}, predicted: {pred_loss:.6f}, continued_last: {pred_train_losses[-1]:.6f}")
    except Exception as e:
        print(f"[{idx}] ????: {e}")

# ========== ?? ========== 
joblib.dump(random_losses, os.path.join(output_dir, "random_initial_losses.pkl"))
joblib.dump(predicted_losses, os.path.join(output_dir, "predicted_initial_losses.pkl"))
joblib.dump(continued_loss_traces, os.path.join(output_dir, "continued_training_losses.pkl"))
joblib.dump(random_init_loss_traces, os.path.join(output_dir, "random_initial_training_losses.pkl"))
