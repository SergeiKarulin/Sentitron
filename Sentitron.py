import torch

class Sentitron:
    def __init__(self, size_of_cortex_layer=100, neuron_synapse_forming_area_size=7, 
                 mediatorDoseFromFire=2, mediatorDoseFromTouch=10, mediatorDecaySpeed=0.25, 
                 potentialDecaySpeed=0.1, activationPotential=3, synapseStrengthRange=1):
        # Parameters
        self.mediatorDoseFromFire = mediatorDoseFromFire
        self.mediatorDoseFromTouch = mediatorDoseFromTouch
        self.mediatorDecaySpeed = mediatorDecaySpeed
        self.potentialDecaySpeed = potentialDecaySpeed
        self.activationPotential = activationPotential
        self.synapseStrengthRange = synapseStrengthRange
        self.neuronSynapseFormingAreaSize = neuron_synapse_forming_area_size
        self.sizeOfCortexLayer = size_of_cortex_layer
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using CUDA!" if torch.cuda.is_available() else "CUDA not available. Using CPU.")

        # Initializing tensors
        self.cortex = torch.zeros(self.sizeOfCortexLayer, self.sizeOfCortexLayer, dtype=torch.float16).to(self.device)
        self.synapses = self.initialize_synapses()
        self.synapse_mediator_amount = torch.zeros_like(self.synapses, dtype=torch.float16).to(self.device)

    def initialize_synapses(self):
        synapses = torch.rand(self.sizeOfCortexLayer, self.sizeOfCortexLayer, 
                              self.sizeOfCortexLayer, self.sizeOfCortexLayer, dtype=torch.float16).to(self.device)
        # Setting up the synapse mask
        center = self.neuronSynapseFormingAreaSize // 2
        mask = torch.zeros_like(synapses, dtype=torch.bool)
        for i in range(self.sizeOfCortexLayer):
            for j in range(self.sizeOfCortexLayer):
                row_start = max(i - center, 0)
                row_end = min(i + center + 1, self.sizeOfCortexLayer)
                col_start = max(j - center, 0)
                col_end = min(j + center + 1, self.sizeOfCortexLayer)
                mask[i, j, row_start:row_end, col_start:col_end] = True
        synapses[~mask] = 0
        return synapses

    def touch(self, touches):
        for x, y in touches:
            self.cortex[x, y] = self.mediatorDoseFromTouch

    def Tick(self):
        mask = self.cortex >= self.activationPotential
        self.synapse_mediator_amount[mask] += self.mediatorDoseFromFire
        self.cortex[mask] -= self.activationPotential
        self.cortex += (self.synapse_mediator_amount * self.synapses).sum(dim=0).sum(dim=0)
        self.cortex[self.cortex < 0] = 0  # We don't need negative potential
        a = 1 - self.mediatorDecaySpeed
        self.synapse_mediator_amount *= (1 - self.mediatorDecaySpeed)
        self.synapse_mediator_amount[self.synapse_mediator_amount < 0] = 0
        self.cortex *= (1 - self.potentialDecaySpeed)
        return((self.cortex >= self.activationPotential * (1 - self.potentialDecaySpeed)).count_nonzero())
