import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import os

# criando uma rede com 5 neuronios, 1 camada oculta com 5 neuronios e 3 outputs
# na entrada ela vai receber os seguintes dados: obstaculo distancia, obstaculo largura, obstaculo altura, velocidade do obstaculo, altura do dino

class RedeNeural(nn.Module):
    def __init__(self):
        super(RedeNeural, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU()
        )
        
        self.out = nn.Linear(5, 3)
        # self.relu = nn.ReLU
    
    def forward(self, x):
        feature = self.features(x)
        # output = self.relu(self.out(feature))
        output = self.out(feature)
        return output


    def save(self, file_name='model.pht'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            # adicionando uma dimensão
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )
        
        pred = self.model(state)
        
        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()