"""Author: Klemen Kubelj
    made on 30.08.2017"""

import numpy as np

class Board:
    def __init__(self, width = 8, height = 7):
        self.width = width
        self.height = height
        self.field = np.zeros((height, width),dtype=int)

    def resize(self, width, height):
        self.width = width
        self.height = height
        self.field = np.zeros((height, width),dtype=int)

    def insert_x(self, stolp):
        for i in np.arange(self.height-1,-1,-1):
            if self.field[i][stolp] == 0:
                self.field[i][stolp] = 1
                break

    def insert_o(self, stolp):
        for i in np.arange(self.height-1, -1, -1):
            if self.field[i][stolp] == 0:
                self.field[i][stolp] = 2
                break


def draw(board):
    for i in range(board.height):
        print('')
        for j in range(board.width):
            if board.field[i][j] == 0: print('_', end=' ')
            if board.field[i][j] == 1: print('x', end=' ')
            if board.field[i][j] == 2: print('o', end=' ')

def P1_move(board):
    s = int(input('Igralec 1 vnese stolpec:'))
    board.insert_x(s)

def P2_move(board):
    s = int(input('Igralec 2 vnese stolpec:'))
    board.insert_o(s)

def game():
    plosca = Board()

    while True:
        draw(plosca)
        P1_move(plosca)
        draw(plosca)
        P2_move(plosca)

game()
