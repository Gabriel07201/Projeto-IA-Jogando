import pygame
import random
import numpy as np

# esse código será o utilizado pela rede neural para jogar

pygame.init()
LARGURA_TELA, ALTURA_TELA = 1000, 800

# variáveis do dino
DINO_LARGURA, DINO_ALTURA = 100, 100
DINO_X, DINO_Y = 100, 500
DINO_GRAVIDADE = 5

# variáveis do obstáculo
OBSTACLE_LARGURA, OBSTACLE_ALTURA = 70, 70
OBS_Y = 530
MAX_OBS = 7
OBS_VEL_INICIAL = 5



class Dino:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.largura = DINO_LARGURA
        self.altura = DINO_ALTURA
        self.img = pygame.transform.scale(pygame.image.load("img_dinossauro.png"), (100, 100)).convert_alpha()
        self.pulo = False
        self.vel = DINO_GRAVIDADE
    
    def pular(self):
        if self.pulo == False:
            self.y -= 250
            self.pulo = True
    
    def aplicar_gravidade(self):
        if self.y < 500:
            self.y += self.vel
        else:
            self.pulo = False
    
    def abaixar(self):
        self.y = 500
    
    def desenhar(self, tela):
        tela.blit(self.img, (self.x, self.y))
    
    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Obstaculo:
    def __init__(self, x, y, velocidade):
        self.x = x
        self.y = y
        self.largura = OBSTACLE_LARGURA
        self.altura = OBSTACLE_ALTURA
        self.img = pygame.transform.scale(pygame.image.load('dinoSpritesheet.png').subsurface((32*5, 0, 32, 32)), (OBSTACLE_LARGURA, OBSTACLE_ALTURA)).convert_alpha()
        self.vel = velocidade
        self.passado = False
    
    def desenhar(self, tela):
        tela.blit(self.img, (self.x, self.y))
    
    def mover(self):
        self.x -= self.vel
    
    def get_mask(self):
        return pygame.mask.from_surface(self.img)
    
    def colidir(self, dino):
        dino_mask = dino.get_mask()
        obs_mask = self.get_mask()
        
        offset = (self.x - dino.x, self.y - round(dino.y))
        
        colisao = dino_mask.overlap(obs_mask, offset)
        
        if colisao:
            return True
        
        return False



class GameAI:
    def __init__(self) -> None:
        self.run = True
        self.display = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
        self.clock = pygame.time.Clock()
        self.pontuacao_maxima = 0
        self.reset()
    
    def reset(self):
        self.dino = Dino(DINO_X, DINO_Y)
        self.obstaculos = []
        self.obs_timer = 0
        self.pontuacao_atual = 0
        self.velocidade_obs = OBS_VEL_INICIAL
        self.frame_iteration = 0
        self.obs_passados = 0
    
    def gerar_obs(self):
        self.obstaculos.append(Obstaculo(1000 + random.randint(100, 500), OBS_Y, self.velocidade_obs))
    
    def mover_obs(self):
        for obs in self.obstaculos:
            obs.mover()
            if self.dino.x > obs.x and obs.passado == False:
                obs.passado = True
                self.obs_passados += 1
            if obs.x < -100:
                self.obstaculos.remove(obs)
    
    def pegar_prox_obs(self):
        next_obs = None
        min_x = float('inf')
        for obs in self.obstaculos:
            if obs.x > self.dino.x and obs.x < min_x:
                next_obs = obs
                min_x = obs.x
        return next_obs
        
    
    def colidiu(self):
        for obs in self.obstaculos:
            if obs.colidir(self.dino):
                return True
        return False
    
    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
                pygame.quit()
                quit()
        
        # movimentação do dino
        self.dino.aplicar_gravidade()
        self._move(action)
        
        # gerar e mover obstáculos
        if self.pontuacao_atual % 100 == 0:
            self.obs_timer = 0
            if len(self.obstaculos) < MAX_OBS:
                self.gerar_obs()
        self.mover_obs()
        
        # aumentar velocidade dos obstáculos
        if self.pontuacao_atual % 1000 == 0:
            self.velocidade_obs += 1
            
        
        # check colisão
        recompensa = 0
        game_over = False
        if self.colidiu():
            self.run = False
            game_over = True
            recompensa = -100
            return recompensa, game_over, self.pontuacao_atual
        else:
            recompensa = 10 * self.obs_passados
            self.pontuacao_atual += round(1 * (self.velocidade_obs / 10), 0)
        
        self._update_ui()
        self.clock.tick(60)
        
        return recompensa, game_over, self.pontuacao_atual
        
    
    def _update_ui(self):
        self.display.fill((255, 255, 255))
        for obs in self.obstaculos:
            obs.desenhar(self.display)
        self.dino.desenhar(self.display)
        
        texto_pontos = pygame.font.SysFont('comicsans', 30).render(f'Pontuação: {self.pontuacao_atual}', 1, (0, 0, 0))
        self.display.blit(texto_pontos, (10, 50))
        
        pygame.display.update()
        
    
    def _move(self, action):
        if np.array_equal(action, [1, 0, 0]):
            self.dino.pular()
        elif np.array_equal(action, [0, 1, 0]):
            self.dino.abaixar()
        else:
            pass