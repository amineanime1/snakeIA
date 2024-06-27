import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from utils import empty_images_folder,load_total_games_played, save_total_games_played, load_record, save_record, load_total_trainings, save_total_trainings
from model import Linear_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

def load_model(model_path):
    model = None
    try:
        if os.path.exists(model_path):
            print('Loading model...')
            model = Linear_QNet(11, 256, 3)
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"No model found at {model_path}")

    except Exception as e:
        print(f"Error loading model: {e}")
    return model

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

class Agent:
    
    def __init__(self):
        self.n_games = load_total_games_played()  # Charger le nombre total de parties jouées
        self.n_trainings = load_total_trainings()  # Load the total number of trainings
        self.epsilon = 0 # randomness  
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            
            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            
        ]
        
        return np.array(state, dtype=int)
    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 4000 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 10000) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            prediction = self.model(state0) # this will lead to the forward model method
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move
    
def train():
        total_trainings = load_total_trainings()
        record = load_record()
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        agent = Agent()
        game = SnakeGameAI()
        
        total_trainings += 1
        save_total_trainings(total_trainings)
        # Charger un modèle pré-entraîné s'il existe
        
        model_path = 'model/model.pth'
        if os.path.exists(model_path):
            print('Loading model...')
            agent.model = load_model(model_path)
        if agent.model is None:
            print('No model found, initializing a new one...')
            record = 0
            total_trainings = 0
            agent.n_games = 0
            save_record(record)
            save_total_trainings(total_trainings)
            save_total_games_played(agent.n_games)
            empty_images_folder()
            agent.model = Linear_QNet(11, 256, 3)
            
        while True:
            # get old state
            state_old = agent.get_state(game)
    
            # get move
            final_move = agent.get_action(state_old)
            
            # perform move and get new state    
            reward, done, score = game.play_step(final_move)
            
            # get new state
            state_new = agent.get_state(game)   
            
            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            
            # remember
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                # train long memory (experience replay) and plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                
                if score > record:
                    record = score
                    agent.model.save()
                    save_record(record)
                    
                total_games = agent.n_games
                
                print('Game', total_games, 'Score', score, 'Record:', record)
                
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / total_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores, total_trainings)
                
                save_total_games_played(agent.n_games)
                save_model(agent.model, model_path)
                save_total_trainings(total_trainings)
                
if __name__ == '__main__':
    train()