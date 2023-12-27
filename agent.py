import torch
import random
import numpy as np
from collections import deque
from game_ai import GameAI
from rede_neural import RedeNeural, QTrainer
from time import time


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = RedeNeural()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, game):
        dino = game.dino
        obstaculo = game.pegar_prox_obs()
        if obstaculo is None:
        # Retorne um estado padrão ou nulo
            return np.array([dino.y, 0, 0, 0, 0], dtype=np.int64)
        else:
            dist = abs(dino.x - obstaculo.x)
            return np.array([dino.y, dist, obstaculo.x, obstaculo.y, obstaculo.vel], dtype=np.int64)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 10 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 40) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    recorde = 0
    start_time = time()
    agent = Agent()
    game = GameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > recorde:
                recorde = score
                agent.model.save()
            end_time = time()
            # tempo de treino em minutos
            tempo_de_treino_minutos = round((end_time - start_time) // 60)
            tempo_de_treino_segundos = round((end_time - start_time) % 60)
            
            print('Game', agent.n_games, 'Score', score, 'Record:', recorde, f'Tempo de treino: {tempo_de_treino_minutos}:{tempo_de_treino_segundos}')
            
if __name__ == '__main__':
    train()