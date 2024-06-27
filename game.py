import pygame
import random
from enum import Enum
from collections import namedtuple
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)


#


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 10000

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
                    
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # if event.type == pygame.KEYDOWN:
            #     if (event.key == pygame.K_LEFT) and self.direction != Direction.RIGHT:
            #         self.direction = Direction.LEFT
            #     elif (event.key == pygame.K_RIGHT) and self.direction != Direction.LEFT:
            #         self.direction = Direction.RIGHT
            #     elif (event.key == pygame.K_UP) and self.direction != Direction.DOWN:
            #         self.direction = Direction.UP
            #     elif (event.key == pygame.K_DOWN) and self.direction != Direction.UP:
            #         self.direction = Direction.DOWN
        
        # Add pathfinding before the snake moves
        path = self.find_path()
        self.follow_path(path)
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
         # Add a negative reward for a move that would result in a collision
        if self._will_collide():
            reward -= 3   
        if self._is_encircled():
            reward -= 5     
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 20
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, point=None):
        if point is None:
            point = self.head
    # check if we hit the border
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
         return True
    # check if we hit ourselves
        if point in self.snake[1:]:
            return True
        return False
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _will_collide(self):
        next_head = self._get_next_head_position(self.direction)
        return next_head in self.snake[1:]

    def _get_next_head_position(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
         x += BLOCK_SIZE
        elif direction == Direction.LEFT:
           x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
          y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        return Point(x, y)

    def _move(self, action):
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
             
        self.direction = new_dir
        
        self.head = self._get_next_head_position(self.direction)
    def _get_free_directions(self):
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        free_directions = 0
        for direction in directions:
            next_head = self._get_next_head_position(direction)
            if not self.is_collision(next_head):
                free_directions += 1
        return free_directions

    def _is_encircled(self):
        return self._get_free_directions() == 1
    def find_path(self):
    # Create a 2D array representing the game field
    # 0 represents a free cell, 1 represents a cell occupied by the snake
        field = [[0 for _ in range(self.w // BLOCK_SIZE)] for _ in range(self.h // BLOCK_SIZE)]
        for pt in self.snake:
            field[int(pt.y // BLOCK_SIZE)][int(pt.x // BLOCK_SIZE)] = 1

        grid = Grid(matrix=field)

        start = grid.node(int(self.head.x // BLOCK_SIZE), int(self.head.y // BLOCK_SIZE))
        end = grid.node(int(self.food.x // BLOCK_SIZE), int(self.food.y // BLOCK_SIZE))

        finder = AStarFinder()
        path, _ = finder.find_path(start, end, grid)

        return path

    def follow_path(self, path):
        if not path:
            return

        next_node = path[0]
        next_point = Point(next_node[0] * BLOCK_SIZE, next_node[1] * BLOCK_SIZE)

        if next_point.x > self.head.x:
            self.direction = Direction.RIGHT
        elif next_point.x < self.head.x:
            self.direction = Direction.LEFT
        elif next_point.y > self.head.y:
            self.direction = Direction.DOWN
        elif next_point.y < self.head.y:
            self.direction = Direction.UP