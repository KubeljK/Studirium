# Import a library of functions called 'pygame'
import pygame
from math import sin, cos, sqrt, pi

# Initialize the game engine
pygame.init()
clock = pygame.time.Clock()

# Define the colors we will use in RGB format
black = (0,   0,   0)
white = (255, 255, 255)

# Set the height and width of the screen
screen_width = 1000
screen_height = 1000
screen = pygame.display.set_mode((screen_width, screen_height))

pygame.display.set_caption("Nihalo_spremenjeno")

print("Hello svet 2")

def game_loop():
    # Loop until the user clicks the close button.
    done = False
    L = 200
    fi0 = 45 * pi / 180
    g = 9.81
    t = 0
    dt = 0.1
    center = [screen_width/2, screen_height/10]

    while not done:

        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        # All drawing code happens after the for loop and but
        # inside the main while done==False loop.

        # Clear the screen and set the screen background
        screen.fill(white)

        fi = fi0*cos(sqrt(g/L)*t)
        t += dt

        pygame.draw.line(screen, black, center,
                         [center[0]+L*sin(fi), center[1]+L*cos(fi)], 2)

        pygame.draw.circle(screen, black, [int(center[0]+L*sin(fi)),
                                           int(center[1]+L*cos(fi))], 10)

        # Go ahead and update the screen with what we've drawn.
        # This MUST happen after all the other drawing commands.
        pygame.display.update()
        clock.tick(60)


# Be IDLE friendly
game_loop()
pygame.quit()
quit()