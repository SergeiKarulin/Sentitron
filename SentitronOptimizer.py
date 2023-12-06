import numpy as np
from Sentitron import Sentitron


def objective_function(simulator, target, cyclesToRun):
    total_firing_cycles = 0
    for _ in range(cyclesToRun):
        tick = simulator.Tick()
        if 0 < tick < simulator.sizeOfCortexLayer ** 2:
            total_firing_cycles += 1
        else:
            break
    return abs(total_firing_cycles - target)

def gradient_descent(simulator, params, learning_rate, iterations, cyclesToRun, target):
    for i in range(iterations):
        # Calculate gradients
        # This part is highly dependent on how Sentitron class is implemented and whether it supports automatic differentiation.
        # You may need a framework like PyTorch or TensorFlow if Sentitron is not currently set up for this.
        gradients = compute_gradients(simulator, params)

        # Update parameters
        for param in params:
            params[param] -= learning_rate * gradients[param]

        # Update the simulator with new parameters
        simulator.reInit(**params)

        # Evaluate objective function
        loss = objective_function(simulator, target, cyclesToRun)
        print(f"Iteration {i}, Loss: {loss}")

        # Early stopping condition or other convergence criteria can be added here
    return params

def compute_gradients(simulator, params, objective_function, cyclesToRun, target, epsilon=1e-4):
    """
    :param simulator: Instance of Sentitron model.
    :param params: Dictionary of current parameters.
    :param objective_function: The objective function to minimize.
    :param cyclesToRun: Number of cycles to run in the simulation.
    :param target: Target value for the objective function.
    :param epsilon: Small value for numerical derivative.
    :return: Dictionary of gradients for each parameter.
    """
    gradients = {}

    # Original loss
    original_loss = objective_function(simulator, target, cyclesToRun)

    for param in params:
        # Save the original value of the parameter
        original_value = params[param]
        # Perturb the parameter value
        simulator.reInit(**{param: original_value + epsilon})      
        # Compute new loss
        new_loss = objective_function(simulator, target, cyclesToRun)
        # Approximate gradient (numerical differentiation). Here we decide the sign of gradient to use for the nex iteration
        gradient = (new_loss - original_loss) / epsilon
        # Store the computed gradient
        gradients[param] = gradient
        # Reset the parameter to its original value
        simulator.reInit(**{param: original_value})

    return gradients

# Initial parameters
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

# Initialize simulator
model = Sentitron(**initial_params)

# Optimization settings
learning_rate = 0.1
iterations = 10
cyclesToRun = 100
target = cyclesToRun*0.5

# Perform optimization
optimized_params = gradient_descent(model, initial_params.__delitem__("sizeOfCortexLayer"), learning_rate, iterations, cyclesToRun, target)