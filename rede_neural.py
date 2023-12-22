from torch import nn
from torch import optim

# criando uma rede com 5 neuronios, 1 camada oculta com 5 neuronios e 2 outputs
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
        
        self.out = nn.Linear(5, 2)
        self.relu = nn.ReLU
    
    def forward(self, x):
        feature = self.features(x)
        output = self.relu(self.out(feature))
        return output


def otimizando_rede(model, n_novas_redes):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # otimizando a melhor rede e gerando novas redes
    