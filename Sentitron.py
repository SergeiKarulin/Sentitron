import torch
import random

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA!")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

def touch(cortex_neurons, x, y):
    global mediatorDoseFromTouch
    cortex_neurons[x,y] = mediatorDoseFromTouch

mediatorDoseFromFire = 1
mediatorDoseFromTouch = 10
mediatorDecaySpeed = 0.25
potentiaDecaySpeed = 0.1
activationPotential = 2
synapseStrengthRange = 4 #Will work as from -2 to 2

maxBeatsForExperiment = 100

theSizeOfCortexLayer = 200

cortex = torch.zeros(theSizeOfCortexLayer, theSizeOfCortexLayer, dtype=torch.float16)
cortex = cortex.to(device)

synapses = torch.rand(theSizeOfCortexLayer, theSizeOfCortexLayer, theSizeOfCortexLayer, theSizeOfCortexLayer, dtype=torch.float16)
synapses = synapses.to(device)
synapses = synapses*synapseStrengthRange - (synapseStrengthRange/2)

synapse_mediator_amount = torch.zeros(theSizeOfCortexLayer, theSizeOfCortexLayer, theSizeOfCortexLayer, theSizeOfCortexLayer, dtype=torch.float16)
synapse_mediator_amount = synapse_mediator_amount.to(device)  
    
touch(cortex, 0, 0)
beat = 0

while ((cortex >= activationPotential*(1-potentiaDecaySpeed)).any()) and (beat < maxBeatsForExperiment):
    mask = cortex >= activationPotential
    synapse_mediator_amount[mask] += mediatorDoseFromFire
    cortex[mask] -= activationPotential
    cortex += synapse_mediator_amount.sum(dim=0).sum(dim=0)*synapses.sum(dim=0).sum(dim=0)
    cortex[cortex < 0] = 0 #We cdon't need negative potential
    synapse_mediator_amount *= (1 - mediatorDecaySpeed)
    cortex *= (1 - potentiaDecaySpeed)
    print(cortex)
    beat += 1
    
#The next step is to limit the access areas for axons