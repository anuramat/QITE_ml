import math
import cmath
import numpy as np




class Ansatz():

    def __init__(self, n_qubits, n_layers, nonzero_qubit_index = 0):
        self.n_qubits = n_qubits
        self.psi_in = np.zeros(2**n_qubits)
        self.psi_in[nonzero_qubit_index] = 1 # TODO does it matter? am I doing this wrong?
        self.n_layers = n_layers
        self.n_theta = (n_layers + 1) * n_qubits

        # construct entangling matrix
        controlled_x = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
        entangling_matrix = np.identity(2 ** n_qubits)
        for q in range(n_qubits - 1):
            tmp = np.kron(
                np.kron(np.identity(2 ** (n_qubits - 2 - q)), controlled_x),
                np.identity(2 ** q),
            )
            entangling_matrix = tmp @ entangling_matrix  # The order is important!!!

        # convert entangling matrix to a mask
        self.entangling_mask = (
            entangling_matrix @ np.array([i for i in range(2 ** n_qubits)])
        ).astype(int)

    def rotate_y(self, theta, q, vec):
        mask = [i ^ (1 << q) for i in range(len(vec))]
        sgn = np.array([1 if (i & (1 << q)) == 0 else -1 for i in range(len(vec))])
        return math.cos(0.5 * theta) * vec + math.sin(0.5 * theta) * vec[mask] * sgn[mask]

    def generate_pairs(self, n_pairs=1):
        psi = self.psi_in.copy()
        psi_list = []
        theta_list = []
        for pair in range(n_pairs):
            theta = math.pi * (2 * np.random.rand(self.n_theta) - 1)
            psi = self.apply_ansatz(theta)
            psi_list.append(psi)
            theta_list.append(theta)
        return [psi_list, theta_list]


    def apply_ansatz(self, theta):
        psi = self.psi_in.copy()
        for ilayer in range(self.n_layers):
            for iq in range(self.n_qubits):

                iang = ilayer * self.n_qubits + iq

                psi = self.rotate_y(theta[iang], iq, psi)

            if ilayer != self.n_layers:
                psi = psi[self.entangling_mask]
        return psi