class Neuron:
    def __init__(self, activation_threshold, excitation_step):
        self.activation_threshold = activation_threshold  # The threshold at which the neuron activates
        self.current_excitation = 0  # Current level of excitation
        self.excitation_step = excitation_step  # Step by which excitation increases each tact

    def activate(self):
        # Function to activate the neuron and send signals to connected synapses
        # This will be expanded later
        self.current_excitation = 0  # Reset excitation level to 0 upon activation

    def process_tact(self):
        # Function to process each tact
        self.increase_excitation()

    def increase_excitation(self):
        # Increase the current excitation towards the threshold
        self.current_excitation += self.excitation_step
        if self.current_excitation >= self.activation_threshold:
            self.activate()
