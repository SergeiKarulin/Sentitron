import torch

class Sentitron:
    def __init__(self, sizeOfCortexLayer=100, neuronSynapseFormingAreaSize=7, 
                 mediatorDecaySpeed=0.25, potentialDecaySpeed=0.1, activationPotential=3, 
                 synapseStrengthRange=1, mediatorDoseFromFire=2, mediatorDoseFromTouch=10):
        
        self.sizeOfCortexLayer = sizeOfCortexLayer #6Gb GPU ~ 150, 16Gb GPU ~ 210-215
        self.mediatorDoseFromFire = mediatorDoseFromFire #The amount of mediator passed to the synapse on neuron activation
        self.mediatorDoseFromTouch = mediatorDoseFromTouch #The amount of mediator we pass to the synapse when touch it
        self.mediatorDecaySpeed = mediatorDecaySpeed #The speed of difusion in synapses
        self.potentialDecaySpeed = potentialDecaySpeed #The speed how neuron loses tension
        self.activationPotential = activationPotential
        self.synapseStrengthRange = synapseStrengthRange #Possible synapse weights: 2 = [-1;1], 5 = [-2.5;2.5]
        self.neuronSynapseFormingAreaSize = neuronSynapseFormingAreaSize #The size of an area whnere neuron creates axonal connections (5 = 5x5), better to keep it even

        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using CUDA!" if torch.cuda.is_available() else "CUDA not available. Using CPU.")

        self.cortex = torch.zeros(self.sizeOfCortexLayer, self.sizeOfCortexLayer, dtype=torch.float16).to(self.device)
        self.synapses = self.initialize_synapses()
        self.synapse_mediator_amount = torch.zeros_like(self.synapses, dtype=torch.float16).to(self.device)

    def reInit(self, mediatorDecaySpeed, potentialDecaySpeed, 
                 activationPotential, synapseStrengthRange, mediatorDoseFromFire, mediatorDoseFromTouch):
        #Changes meta parameters keeping the size unchanged
        self.mediatorDoseFromFire = mediatorDoseFromFire
        self.mediatorDoseFromTouch = mediatorDoseFromTouch
        self.mediatorDe0caySpeed = mediatorDecaySpeed
        self.potentialDecaySpeed = potentialDecaySpeed
        self.activationPotential = activationPotential
        self.synapseStrengthRange = synapseStrengthRange

        self.cortex.zero_()
        self.synapses = self.initialize_synapses()
        self.synapse_mediator_amount.zero_()
    
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
        self.synapse_mediator_amount *= (1 - self.mediatorDecaySpeed)
        self.synapse_mediator_amount[self.synapse_mediator_amount < 0] = 0 # We don't need negative mediator amount
        self.cortex *= (1 - self.potentialDecaySpeed)
        return((self.cortex >= self.activationPotential * (1 - self.potentialDecaySpeed)).count_nonzero())
