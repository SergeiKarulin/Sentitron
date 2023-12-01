import torch
import random

class Sentitron:
    def __init__(self, size, max_connections = 3, min_connections = 1, loop_probability = 0, tranquility = 1.1, action_potential = 1, mediator_decay_rate = 0.25, fire_strength = 2, polarization_decay_rate = 0.2):
        self.size = size
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.loop_probability = loop_probability
        
        self.tranquility = tranquility
        
        self.action_potential = action_potential
        self.mediator_decay_rate = mediator_decay_rate
        self.fire_strength = fire_strength
        self.touch_strength = 5
        self.polarization_decay_rate = polarization_decay_rate
        
        self.neuron_layer = torch.zeros(size, size)
        self.synapse_weights = torch.zeros(size, size, size, size)
        self.synapse_mediators = torch.zeros(size, size, size, size)
        self.init_synapses(self.max_connections, self.min_connections, self.loop_probability)

    def init_synapses(self, max_connections, min_connections, loop_probability, std_dev=1.0):
        x_coords, y_coords = torch.meshgrid(torch.arange(self.size), torch.arange(self.size))
        x_diff = x_coords.unsqueeze(2).unsqueeze(3) - x_coords.unsqueeze(0).unsqueeze(1)
        y_diff = y_coords.unsqueeze(2).unsqueeze(3) - y_coords.unsqueeze(0).unsqueeze(1)
        distances = torch.sqrt(x_diff**2 + y_diff**2)
        gaussian_prob = torch.exp(-distances**2 / (2 * std_dev**2))
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
                    self.synapse_weights[i, j, x, y] = 2 * torch.rand(1).item() - self.tranquility

    def recalculate_depolarization(self):
        input_sum = torch.sum(self.synapse_weights * self.synapse_mediators, dim=(0, 1))
        self.neuron_layer += input_sum

        firing_neurons = self.neuron_layer >= self.action_potential
        
        firing_indices = firing_neurons.nonzero()
        if firing_indices.numel() > 0:
            self.fire(firing_indices)

    def fire(self, firing_indices):
        self.neuron_layer[firing_indices[:, 0], firing_indices[:, 1]] -= self.action_potential
        firing_mask = torch.zeros_like(self.synapse_weights, dtype=torch.bool)
        firing_mask[firing_indices[:, 0], firing_indices[:, 1], :, :] = True
        total_out = torch.sum(torch.abs(self.synapse_weights) * firing_mask, dim=(2, 3))

        total_out[total_out == 0] = 1

        mediator_update = (self.fire_strength / total_out.unsqueeze(2).unsqueeze(3)) * firing_mask
        self.synapse_mediators += mediator_update

    def diffuse(self):
        self.synapse_mediators -= self.mediator_decay_rate
        self.synapse_mediators.clamp_(0)
        
    def decay_depolarization(self):
        positive_mask = self.neuron_layer > 0
        negative_mask = self.neuron_layer < 0
        self.neuron_layer[positive_mask] = torch.clamp(self.neuron_layer[positive_mask] - self.polarization_decay_rate, min=0)
        self.neuron_layer[negative_mask] = torch.clamp(self.neuron_layer[negative_mask] + self.polarization_decay_rate, max=0)

    def beat(self):
        self.recalculate_depolarization()
        self.decay_depolarization()
        self.diffuse()
        return self.neuron_layer
    
    def touch(self, x, y):
        self.neuron_layer[x, y] += self.touch_strength