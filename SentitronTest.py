from Sentitron import Sentitron

#Initialize the Simulation
sizeOfCortexLayer = 150
neuronSynapseFormingAreaSize = 25 #Big steps are ok
mediatorDoseFromFire = 0.1
mediatorDoseFromTouch = 100 #Doesn't matter
mediatorDecaySpeed = 0.05 #Minimal steps only
potentialDecaySpeed = 0.1
activationPotential = 5
synapseStrengthRange = 4
simulator = Sentitron(sizeOfCortexLayer, neuronSynapseFormingAreaSize, mediatorDoseFromFire, mediatorDoseFromTouch, mediatorDecaySpeed, potentialDecaySpeed, activationPotential, synapseStrengthRange)

#Run the Simulation
simulator.touch([(10, 10), (75,75)])  
for _ in range(1000): 
    tick = simulator.Tick()
    if (tick > 0) and (tick<sizeOfCortexLayer*sizeOfCortexLayer): print(str(simulator.Tick()) + " neurons will fire on the next tick") 
    else: break
    
#TODO: Next step is to prepare investigation of parameters that allow oscilations#