# This code is part of the variational-lse-solver library.
#
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This file contains the main access point for using the variational LSE solver.
"""

import numpy as np
import torch
import pennylane as qml
import warnings
from typing import Callable
from tqdm import tqdm
import sys

from .utils import mode_init, mode_pauli_validation, mode_unitary_validation, mode_circuit_validation, mode_matrix_validation
from .utils import method_init, method_direct_validate, method_hadamard_validate, method_overlap_validate, method_coherent_validate
from .utils import init_imaginary_flag, init_imaginary_flag_dummy
from .circuits import dynamic_circuit

from .cost_function import CostFunction
from .cost_function.cost_function_types import CostFunctionMode, CostFunctionMethod, CostFunctionLoss


class VarLSESolver:
    def __init__(
        self,
        a: np.ndarray | list[str | np.ndarray | Callable],
        b: np.ndarray | Callable,
        coeffs: list[float | complex] = None,
        ansatz: Callable = None,
        weights: tuple[int, ...] | np.ndarray = None,
        method: str = "direct",
        local: bool = False,
        lr: float = 0.01,
        steps: int = 10000,
        epochs: int = 1,
        threshold: float = 1e-4,
        abort: int = 500,
        seed: int = None,
        data_qubits: int = 0,
    ):
        mode: CostFunctionMode = mode_init(a)
        self.data_qubits = mode.mode_dependent_value(
            pauli=mode_pauli_validation,
            unitary=mode_unitary_validation,
            circuit=mode_circuit_validation,
            matrix=mode_matrix_validation,
        )(a, coeffs, b, data_qubits)

        loss: CostFunctionLoss = (
            CostFunctionLoss.LOCAL if local else CostFunctionLoss.GLOBAL
        )
        method: CostFunctionMethod = method_init(method)
        method.method_dependent_value(
            direct=method_direct_validate,
            hadamard=method_hadamard_validate,
            overlap=method_overlap_validate,
            coherent=method_coherent_validate,
        )(mode, loss, b)

        np.random.seed(seed)

        self.ansatz, self.weights, self.dynamic_circuit, self.epochs = (
            self.init_ansatz_and_weights(ansatz, weights, self.data_qubits, epochs)
        )

        imaginary = mode.mode_dependent_value(
            pauli=init_imaginary_flag,
            unitary=init_imaginary_flag,
            circuit=init_imaginary_flag,
            matrix=init_imaginary_flag_dummy,
        )(coeffs)

        self.lr = lr
        self.opt = torch.optim.Adam([{"params": self.weights}], lr=lr)
        if 0 > threshold > 1:
            raise ValueError("The `threshold` has to be in (0.0, 1.0).")
        self.steps = steps
        self.threshold = threshold
        self.abort = abort

        self.cost_function = CostFunction(
            a, coeffs, b, self.ansatz, mode, method, loss, self.data_qubits, imaginary
        )

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        best_weights = self.weights.detach().numpy()
        for epoch in range(self.epochs):
            best_loss, best_step = 1.0, 0
            if 0 < epoch:
                print("Increasing circuit depth.", flush=True)
                new_weights = np.random.uniform(
                    low=0.0, high=2 * np.pi, size=(1, self.weights.shape[1])
                )
                weights = np.concatenate(
                    (
                        best_weights,
                        np.stack(
                            (
                                new_weights,
                                np.zeros((1, self.weights.shape[1])),
                                -new_weights,
                            ),
                            axis=2,
                        ),
                    )
                )
                self.weights = torch.tensor(weights, requires_grad=True)
                self.opt = torch.optim.Adam([{"params": self.weights}], lr=self.lr)

            pbar = tqdm(range(self.steps), desc=f"Epoch {epoch+1}/{self.epochs}: ", file=sys.stdout)
            for step in pbar:
                self.opt.zero_grad()
                loss = self.cost_function.cost(self.weights)
                if loss.item() < best_loss and abs(loss.item() - best_loss) > 0.1 * self.threshold:
                    best_loss = loss.item()
                    best_step = step
                    best_weights = self.weights.detach().numpy()
                if loss.item() < self.threshold:
                    pbar.close()
                    print(f"Loss of {loss.item():.10f} below stopping threshold.\nReturning solution.", flush=True)
                    return self.evaluate(best_weights), best_weights
                if step - best_step >= self.abort:
                    pbar.close()
                    print(
                        f"Loss has not improved in last {self.abort} steps.\nReturning best solution.",
                        flush=True,
                    )
                    break
                pbar.set_postfix(
                    {
                        "best loss": best_loss,
                        "last improvement in step": best_step,
                        "loss": loss.item(),
                    }
                )
                loss.backward()
                self.opt.step()
        return self.evaluate(best_weights), best_weights

    def evaluate(self, weights: np.array) -> np.array:
        return self.qnode_evaluate_x()(weights).detach().numpy()

    def qnode_evaluate_x(self) -> Callable:
        dev = qml.device("default.qubit", wires=self.data_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit_evolve_x(weights):
            self.ansatz(weights)
            return qml.probs()

        return circuit_evolve_x

    @staticmethod
    def init_ansatz_and_weights(
        ansatz: Callable,
        weights: tuple[int, ...] | np.ndarray,
        data_qubits: int,
        epochs: int,
    ) -> tuple[Callable, torch.tensor, bool, int]:
        if ansatz is None:
            if weights is not None:
                warnings.warn("No explicit `ansatz` was selected, provided `weights` will be ignored.")
            weights = np.random.uniform(low=0.0, high=2 * np.pi, size=(1, data_qubits, 3))
            return dynamic_circuit, torch.tensor(weights, requires_grad=True), True, epochs
        if not callable(ansatz):
            raise ValueError("The provided `ansatz` has to be Callable.")
        if epochs > 1:
            warnings.warn("Explicit `ansatz` was provided, `epochs` argument will be ignored.")
        if isinstance(weights, tuple):
            weights = np.random.uniform(low=0.0, high=2 * np.pi, size=weights)
            return ansatz, torch.tensor(weights, requires_grad=True), False, 1
        elif isinstance(weights, np.ndarray):
            return ansatz, torch.tensor(weights, requires_grad=True), False, 1
        else:
            raise ValueError(
                "The `weights` have to be provided either explicitly as np.ndarray, or as a tuple indicating the shape."
            )

    def test_random_initialized_loss(self):
        """
        Return loss for current randomly initialized weights.
        """
        self.opt.zero_grad()
        loss = self.cost_function.cost(self.weights)
        return loss, self.weights.detach().numpy()

    def test_prediction_loss(self, predicted_weight: np.ndarray):
        """
        Set the weight as the predicted one and return its loss.
        """
        self.weights = torch.tensor(predicted_weight, requires_grad=True)
        self.opt = torch.optim.Adam([{"params": self.weights}], lr=self.lr)
        self.opt.zero_grad()
        loss = self.cost_function.cost(self.weights)
        return loss

    def continue_training_from_predicted(self, predicted_weight: np.ndarray, N: int = 100) -> list[float]:
        """
        Use predicted weights to continue training for N steps.
        """
        self.weights = torch.tensor(predicted_weight, requires_grad=True)
        self.opt = torch.optim.Adam([{"params": self.weights}], lr=self.lr)
        losses = []
        for step in range(N):
            self.opt.zero_grad()
            loss = self.cost_function.cost(self.weights)
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        return losses
