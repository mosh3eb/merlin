import torch

from merlin import OutputMappingStrategy


class FeedForward(torch.nn.Module):
    def __init__(self, layer1, layer2, layer2not, conditonal_mode: int):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer2not = layer2not
        self.conditional_mode = conditonal_mode


    def forward(self, x):
        probs, amplitudes = self.layer1(x, return_amplitudes=True)
        keys = self.layer1.computation_process.simulation_graph.mapped_keys
        layer_1_idx_not, layer_1_idx = self.indices_by_value(keys, self.conditional_mode)
        prob1_not = probs[:, layer_1_idx_not].sum(dim=1)
        prob1 = probs[:, layer_1_idx].sum(dim=1)
        keys_2 = self.layer2.computation_process.simulation_graph.mapped_keys
        match_idx = self.match_indices(keys, keys_2, self.conditional_mode, k_value=1)
        keys_2_not = self.layer2not.computation_process.simulation_graph.mapped_keys
        match_idx_not = self.match_indices(keys, keys_2_not, self.conditional_mode, k_value=0)


        self.layer2.computation_process.input_state = amplitudes[:, match_idx]
        self.layer2not.computation_process.input_state = amplitudes[:, match_idx_not]
        return prob1 * self.layer2(), prob1_not * self.layer2not()





    def indices_by_value(self, keys, k):
        """
        data : liste de tuples ou liste de listes
        k    : position à vérifier
        """
        # convertir en tenseur PyTorch
        t = torch.tensor(keys)

        # indices où la valeur vaut 0
        idx_0 = torch.nonzero(t[:, k] == 0, as_tuple=True)[0]

        # indices où la valeur vaut 1
        idx_1 = torch.nonzero(t[:, k] == 1, as_tuple=True)[0]

        return idx_0, idx_1


    def match_indices(self, data, data_out, k, k_value):
        """
        data      : liste de tuples (longueur n)
        data_out  : liste de tuples (longueur n-1)
        k         : index de la colonne à retirer
        """
        # Conversion en dictionnaire pour retrouver rapidement les indices de data_out
        out_map = {tuple(row): i for i, row in enumerate(data_out)}

        idx= []

        for i, tup in enumerate(data):
            removed = tup[:k] + tup[k + 1:]  # on enlève l'élément k
            if removed in out_map:
                j = out_map[removed]  # index dans data_out
                if tup[k] == k_value:
                    idx.append(j)

        return torch.tensor(idx)



if __name__ == "__main__":
    import merlin as ML
    import perceval as pcvl
    from perceval.components import BS, PS
    from itertools import chain

    def create_circuit(M):

        circuit = pcvl.Circuit(M)

        def layer_bs(circuit, k, M, j):
            for i in range(k, M - 1, 2):
                theta = pcvl.P(f"theta_{i}_{j}")
                circuit.add(i, BS(theta=theta))

        layer_bs(circuit, 0, M, 0)
        layer_bs(circuit, 1, M, 1)
        layer_bs(circuit, 0, M, 2)
        layer_bs(circuit, 1, M, 3)
        layer_bs(circuit, 0, M, 4)
        for i in range(M):
            phi = pcvl.P(f"phi_{i}")
            circuit.add(i, PS(phi))
        layer_bs(circuit, 0, M, 5)
        layer_bs(circuit, 1, M, 6)
        layer_bs(circuit, 0, M, 7)
        layer_bs(circuit, 1, M, 8)
        layer_bs(circuit, 0, M, 9)
        return circuit

    M = 10
    N = 2
    circuit = create_circuit(M)
    input_state =  [0] * (M-N) + [1] * N
    layer = ML.QuantumLayer(
        input_size=M,
        output_size=None,
        circuit=circuit,
        n_photons=N,
        input_state=input_state,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        input_parameters=["phi"],  # Optional: Specify device
        trainable_parameters=["theta"],
        no_bunching=True,
    )
    input_size_2 = layer.output_size // 2
    input_state_2 = [1] * (N-1) + [0] * (M-N)
    input_state_2_not = [1] * N + [0] * (M-N-1)
    circuit_2 = create_circuit(M-1)
    circuit_2_not = create_circuit(M-1)
    layer_2 = ML.QuantumLayer(
        input_size=0,
        output_size=None,
        circuit=circuit_2,
        n_photons=N-1,
        input_state=input_state_2,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        trainable_parameters=["phi", "theta"],  # Optional: Specify device
        no_bunching=True,
    )

    layer_2_not = ML.QuantumLayer(
        input_size=0,
        output_size=None,
        circuit=circuit_2_not,
        n_photons=N,
        input_state=input_state_2_not,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        trainable_parameters=["phi", "theta"],  # Optional: Specify device
        no_bunching=True,
    )
    ff = FeedForward(layer, layer_2, layer_2_not, conditonal_mode=1)
    x = torch.rand(M)
    L = torch.nn.Linear(M, M)
    params = chain(L.parameters(), ff.parameters())
    optimizer = torch.optim.Adam(params)
    for _ in range(10):

        result = (ff(L(x))[0]**2).sum()
        print(result)
        result.backward()
        print(L.weight.grad)
        optimizer.step()
        optimizer.zero_grad()










