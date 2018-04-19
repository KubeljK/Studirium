import pygame
from PIL import Image
import time
import random


pygame.init()


# Windows size.
display_width = 800
display_height = 600

# Basic colors.
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)

# Size of the screen.
gameDisplay = pygame.display.set_mode((display_width, display_height))

# Title.
pygame.display.set_caption("A bit Racey")

clock = pygame.time.Clock()


def make_thing(thingx, thingy, thingw, thingh, color):
    """Make a box in position [thingx, thingy] of dimensions thingw*thingh."""
    pygame.draw.rect(gameDisplay, color, [thingx, thingy, thingw, thingh])


def things_dodged(count):
    """Display the count of dodged boxes in the top-left corner."""
    font = pygame.font.SysFont(None, 25)
    text = font.render("Dodged: "+str(count), True, black)
    gameDisplay.blit(text, (0, 0))


class Car:
    """A class for car objects in our game."""

    def __init__(self, pic_loc="slike/traktor.png"):
        self.img = pygame.image.load(pic_loc)
        im = Image.open(pic_loc)
        self.width, self.height = im.size

    def draw(self, loc):
        """Draw the car at the set location."""
        gameDisplay.blit(self.img, loc)


def text_objects(text, font):
    """Parse the text box for message_display function."""
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()


def crash_message():
    """Display 'You crashed!' on the screen."""
    message_display('You crashed!')


def message_display(text):
    """Display the message text on the screen."""
    largeText = pygame.font.Font('freesansbold.ttf', 115)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width/2), (display_height/2))
    gameDisplay.blit(TextSurf, TextRect)
    pygame.display.update()
    time.sleep(2)
    game_loop()


def game_intro():

    intro = True

    while intro:
        for event in pygame.event.get():
            print(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        gameDisplay.fill(white)
        largeText = pygame.font.Font('freesansbold.ttf', 115)
        TextSurf, TextRect = text_objects("The GAME", largeText)
        TextRect.center = ((display_width/2), (display_height/2))
        gameDisplay.blit(TextSurf, TextRect)
        pygame.display.update()
        clock.tick(15)


# The loop of the game itself.
def game_loop():
    """The main loop of the game."""

    # Starting car location.
    car_x = (display_width * 0.45)
    car_y = (display_height * 0.8)
    x_change = 0
    y_change = 0

    car1 = Car()

    # The first box.
    thing_startx = random.randrange(0, display_width)
    thing_starty = -600
    thing_speed = 7
    thing_width = 100
    thing_height = 100

    # Number of dodged boxes.
    count = 0

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # Move left and right.
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_change = -5
                if event.key == pygame.K_RIGHT:
                    x_change = 5

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    x_change = 0

            # Move up and down.
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    y_change = -5
                if event.key == pygame.K_DOWN:
                    y_change = 5

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    y_change = 0

        car_x += x_change
        car_y += y_change

        # Draw the background and the car itself.
        gameDisplay.fill(white)

        # Make a box.
        make_thing(thing_startx, thing_starty,
                   thing_width, thing_height, black)
        thing_starty += thing_speed
        car1.draw((car_x, car_y))
        things_dodged(count)

        # Check if crashed with box.
        if thing_starty + thing_height > car_y and thing_starty < car_y + car1.height:
            if thing_startx + thing_width > car_x and thing_startx < car_x + car1.width:
                crash_message()

        # Boundaries.
        if car_x > display_width - car1.width or car_x < 0:
            crash_message()

        if car_y > display_height - car1.height or car_y < 0:
            crash_message()

        if thing_starty > display_height:
            thing_starty = 0 - thing_height
            thing_startx = random.randrange(0, display_width)
            count += 1

        # Update the display.
        pygame.display.update()
        clock.tick(60)  # fps


# game_intro()
game_loop()
pygame.quit()
quit()
