import torch
import random

class NeuralNet:
    def __init__(self, size, max_connections=3, min_connections=1, loop_probability=0.1):
        self.size = size
        self.action_potential = 2.0
        self.decay_rate = 0.5
        self.fire_strength = 2
        self.neuron_layer = torch.zeros(size, size)
        self.synapse_weights = torch.zeros(size, size, size, size)
        self.synapse_mediators = torch.zeros(size, size, size, size)
        self.init_synapses(max_connections, min_connections, loop_probability)
        
        # Initialize a random neuron with depolarization value of 10
        random_neuron_x = random.randint(0, size - 1)
        random_neuron_y = random.randint(0, size - 1)
        self.neuron_layer[random_neuron_x, random_neuron_y] = 111

    def init_synapses(self, max_connections, min_connections, loop_probability, std_dev=1.0):
        # Create a meshgrid for neuron coordinates
        x_coords, y_coords = torch.meshgrid(torch.arange(self.size), torch.arange(self.size))

        # Calculate distances between each pair of neurons
        x_diff = x_coords.unsqueeze(2).unsqueeze(3) - x_coords.unsqueeze(0).unsqueeze(1)
        y_diff = y_coords.unsqueeze(2).unsqueeze(3) - y_coords.unsqueeze(0).unsqueeze(1)
        distances = torch.sqrt(x_diff**2 + y_diff**2)

        # Apply Gaussian function to distances
        gaussian_prob = torch.exp(-distances**2 / (2 * std_dev**2))

        # Make sure self-connection probability adheres to the loop_probability
        loop_indices = torch.arange(self.size)
        gaussian_prob[loop_indices, loop_indices, loop_indices, loop_indices] *= loop_probability

        for i in range(self.size):
            for j in range(self.size):
                # Determine potential connections based on Gaussian probabilities
                probs = gaussian_prob[i, j].flatten()
                connections = random.randint(min_connections, max_connections)
                synapse_indices = torch.multinomial(probs, connections, replacement=False)

                for idx in synapse_indices:
                    x, y = idx // self.size, idx % self.size
                    self.synapse_weights[i, j, x, y] = torch.rand(1).item()


    def recalculate_depolarization(self):
        input_sum = torch.sum(self.synapse_weights * self.synapse_mediators, dim=(0, 1))
        self.neuron_layer += input_sum
        firing_neurons = self.neuron_layer >= self.action_potential
        if firing_neurons.any():
            self.fire(firing_neurons)

    def fire(self, firing_neurons):
        indices = firing_neurons.nonzero(as_tuple=True)
        for x, y in zip(*indices):
            self.neuron_layer[x, y] -= self.action_potential
            out_synapses = self.synapse_weights[x, y, :, :]
            total_out = torch.sum(out_synapses > 0, dtype=torch.float32)
            if total_out > 0:
                self.synapse_mediators[x, y, :, :] += (self.fire_strength / total_out) * (out_synapses > 0)


    def diffuse(self):
        self.synapse_mediators -= self.decay_rate
        self.synapse_mediators.clamp_(0)

    def beat(self):
        self.recalculate_depolarization()
        self.diffuse()
        return self.neuron_layer

# Initialize the neural network
neural_net = NeuralNet(size=10)

# Run the beat function every 0.5 seconds
import time
for _ in range(10):  # Run for 10 beats as an example
    neural_net.beat()
    print("Updated Neuron Layer:\n", neural_net.neuron_layer)
    time.sleep(0.5)
