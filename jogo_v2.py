import pygame
import random

pygame.font.init()

# variáveis do jogo
LARGURA_TELA, ALTURA_TELA = 1000, 800
TELA = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))

# variáveis do dino
DINO_IMG = pygame.transform.scale(pygame.image.load("img_dinossauro.png"), (100, 100)).convert_alpha()
DINO_LARGURA, DINO_ALTURA = 100, 100
DINO_X, DINO_Y = 100, 500
DINO_GRAVIDADE = 5

# variáveis do obstáculo
OBSTACLE_LARGURA, OBSTACLE_ALTURA = 70, 70
OBSTACLE_IMG = pygame.transform.scale(pygame.image.load('dinoSpritesheet.png').subsurface((32*5, 0, 32, 32)), (OBSTACLE_LARGURA, OBSTACLE_ALTURA)).convert_alpha()
OBS_Y = 530
MAX_OBS = 7
OBS_VEL_INICIAL = 5



class Dino:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.largura = DINO_LARGURA
        self.altura = DINO_ALTURA
        self.img = DINO_IMG
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
        self.img = OBSTACLE_IMG
        self.vel = velocidade
    
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

def desenhar_tela(tela, dino, obstaculos, pontuacao):
    tela.fill((255, 255, 255))
    for obs in obstaculos:
        obs.desenhar(tela)
    dino.desenhar(tela)
    
    texto_pontos = pygame.font.SysFont('comicsans', 30).render(f'Pontuação: {pontuacao}', 1, (0, 0, 0))
    tela.blit(texto_pontos, (10, 50))
    
    pygame.display.update()
    


def main():
    run = True
    clock = pygame.time.Clock()
    dino = Dino(DINO_X, DINO_Y)
    obstaculos = []
    obs_timer = 0
    pontuacao = 0
    velocidade_obs = OBS_VEL_INICIAL
    while run:
        clock.tick(60)
        obs_timer += 1
        # arrumar pontuação
        pontuacao += round(1 * (velocidade_obs / 5), 0)
        
        # loop responsável por manter a janela aberta e fechar
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        
        # Pulo
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            dino.pular()
        # Agachar
        if keys[pygame.K_DOWN]:
            dino.abaixar()
        
        dino.aplicar_gravidade()
        
        if pontuacao % 100 == 0:
            obs_timer = 0
            if len(obstaculos) < MAX_OBS:
                obstaculos.append(Obstaculo(1000 + random.randint(100, 500), OBS_Y, velocidade_obs))
        
        for obs in obstaculos:
            obs.mover()
            if obs.x < -100:
                obstaculos.remove(obs)
            if obs.colidir(dino):
                run = False
                break
        
        if pontuacao % 1000 == 0:
            velocidade_obs += 1
        
        desenhar_tela(TELA, dino, obstaculos, pontuacao)
        pygame.display.update()
    
    pygame.quit()


if __name__ == "__main__":
    main()