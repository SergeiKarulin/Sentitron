import torch
import numpy as np
import string

class SentytronNetwork:
    def __init__(self, symbols, inner_layer_size, max_synapses):
        # 1. Initialize parameters
        self.symbols = symbols
        self.input_output_size = len(symbols)
        self.inner_layer_size = inner_layer_size
        self.decay_rate = torch.rand(1).item()
        self.excite_threshold = 1
        self.self_excitate_rate = torch.rand(inner_layer_size)
        self.max_synapses = max_synapses

        # 2. Initialize neuron layers
        self.input_output_excitations = torch.zeros(self.input_output_size)
        self.inner_excitations = torch.zeros(inner_layer_size)

        # 3. Initialize synapse weights and mediator amounts for input/output to inner layer
        self.io_to_inner_weights = torch.zeros(self.input_output_size, inner_layer_size)
        self.io_to_inner_mediators = torch.zeros(self.input_output_size, inner_layer_size)
        self._init_synapses(self.io_to_inner_weights, self.io_to_inner_mediators)

        # 4. Initialize synapse weights and mediator amounts for inner to input/output layer
        self.inner_to_io_weights = torch.zeros(inner_layer_size, self.input_output_size)
        self.inner_to_io_mediators = torch.zeros(inner_layer_size, self.input_output_size)
        self._init_synapses(self.inner_to_io_weights, self.inner_to_io_mediators)

    def _init_synapses(self, weights, mediators):
        # Initialize synapses with random connections and weights
        for neuron_idx in range(weights.shape[0]):
            connected_neurons = np.random.choice(range(weights.shape[1]), 
                                                 size=np.random.randint(0, self.max_synapses), 
                                                 replace=False)
            weights[neuron_idx, connected_neurons] = torch.rand(len(connected_neurons))
            mediators[neuron_idx, connected_neurons] = torch.randint(0, 2, (len(connected_neurons),))

    # Define other necessary functions (exchange, mediator_diffuse, self_excitate, etc.)

# English and Russian alphabets, digits
english_alphabet = string.ascii_letters  # Contains both lowercase and uppercase English letters
russian_alphabet = ''.join([chr(i) for i in range(1040, 1104)]) + 'Ёё'  # Contains both lowercase and uppercase Russian letters, including Ё and ё
digits = string.digits
# 20 most used special characters
special_characters = '. , % $ + - * / = & @ # ! ? : ; \' " ( )'
# Combining all symbols
symbols = list(english_alphabet + russian_alphabet + digits + special_characters)

# Example usage
inner_layer_size = 1000
max_synapses = 1000

network = SentytronNetwork(symbols, inner_layer_size, max_synapses)
