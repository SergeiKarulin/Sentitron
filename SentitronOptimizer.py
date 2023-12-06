import numpy as np
from Sentitron import Sentitron

class SentitronGradientDescentMetaParameterOptimizer:
    def __init__(self, initial_params, learning_rate, iterations, cyclesToRun, target):
        self.static_params = {
            "sizeOfCortexLayer": initial_params["sizeOfCortexLayer"],
            "neuronSynapseFormingAreaSize": initial_params["neuronSynapseFormingAreaSize"]
        }
        initial_params.pop("sizeOfCortexLayer", None)
        initial_params.pop("neuronSynapseFormingAreaSize", None)

        self.simulator = Sentitron(**self.static_params, **initial_params)
        self.dynamicParams = initial_params
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.cyclesToRun = cyclesToRun
        self.target = target

    def objective_function(self):
        total_firing_cycles = 0
        for _ in range(self.cyclesToRun):
            tick = self.simulator.Tick()
            if 0 < tick < self.simulator.sizeOfCortexLayer ** 2:
                total_firing_cycles += 1
            else:
                break
        return abs(total_firing_cycles - self.target)

    def compute_gradients(self):
        gradients = {}
        original_loss = self.objective_function()

        for param in self.dynamicParams:
            original_value = self.dynamicParams[param]
            self.simulator.reInit(**{param: original_value + 1e-4})
            new_loss = self.objective_function()
            gradient = (new_loss - original_loss) / 1e-4
            gradients[param] = gradient
            self.simulator.reInit(**{param: original_value})

        return gradients

    def optimize(self):
        for i in range(self.iterations):
            gradients = self.compute_gradients()

            for param in self.dynamicParams:
                self.dynamicParams[param] -= self.learning_rate * gradients[param]

            self.simulator.reInit(**self.dynamicParams)
            loss = self.objective_function()
            print(f"Iteration {i}, Loss: {loss}")

        return self.dynamicParams

#Some Example To Try
initial_params = {
    "sizeOfCortexLayer": 210,
    "neuronSynapseFormingAreaSize": 25,
    "mediatorDecaySpeed": 0.05,
    "potentialDecaySpeed": 0.1,
    "activationPotential": 5,
    "synapseStrengthRange": 4,
    "mediatorDoseFromFire": 0.1,
    "mediatorDoseFromTouch": 100
}

optimizer = SentitronGradientDescentMetaParameterOptimizer(
    initial_params=initial_params,
    learning_rate=0.1,
    iterations=10,
    cyclesToRun=100,
    target=100*0.5
)

# Perform optimization
optimized_params = optimizer.optimize()
print ("Done!")
print (optimized_params)