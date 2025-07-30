# PVLS: GNN-based Parameter Initialization for Variational Quantum Linear Solvers

This repository contains the code for the paper:

**"PVLS: A Learning-based Parameter Prediction Technique for Variational Quantum Linear Solvers"**


## 📌 Features
- GNN-based parameter initializer for VQLSs
- Handles signed and directed graphs
- Supports synthetic and real-world sparse matrices
- Fast convergence on VQC optimization

## 🔧 Installation

```bash
conda create -n pvls python=3.10
conda activate pvls
pip install -r requirements.txt

1.dataset preparation
Option 1: Generate it from scratch

python -m variational_lse_solver.tests.test_me2
This script generates synthetic sparse matrices and corresponding optimal VQC parameters (alpha_opt).
Outputs are saved under the data/ directory.

Option 2: Use pre-generated data
If provided, you can directly use the contents in the data/ folder (train/test/realworld).

2.train the model on synthetic linear systems and test:

python main.py
Model checkpoints, loss curves will be saved automatically.

3.Evaluate on test sets and compare with baseline initializations:

python -m variational_lse_solver.tests.evaluate
python baseline.py
This generates:

Initial loss comparison (PVLS vs Random)

Training loss convergence curves
./vls_baseline contains baseline curves(Random, PCA, MinNorm)

4.📄 License
This repository is licensed under the MIT License.

5. Contact
Youla Yang
School of Informatics, Computing, and Engineering
Indiana University Bloomington
📧 yangyoul@iu.edu