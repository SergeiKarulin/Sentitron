import pygame
from Sentitron import Sentitron

def draw_grid(screen, net, cell_size=20):
    for i in range(net.size):
        for j in range(net.size):
            depolarization = net.neuron_layer[i, j].item()
            if depolarization > net.action_potential:
                color = (128, 0, 128)  # Purple
            elif depolarization == net.action_potential:
                color = (0, 0, 255)  # Blue
            elif depolarization > 0:
                color = (173, 216, 230)  # Light Blue
            else:
                color = (128, 128, 128)  # Gray
            pygame.draw.rect(screen, color, (j * cell_size, i * cell_size, cell_size, cell_size))

def run_simulation(net):
    pygame.init()
    cell_size = 20
    screen = pygame.display.set_mode((net.size * cell_size, net.size * cell_size))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos[1] // cell_size, event.pos[0] // cell_size
                net.touch(x, y)

        net.beat()
        screen.fill((0, 0, 0))  
        draw_grid(screen, net, cell_size)
        pygame.display.flip()
        clock.tick(2) 

    pygame.quit()

neural_net = Sentitron(size=25)
run_simulation(neural_net)