import pygame
import random

pygame.font.init()

# Window size
WIDTH, HEIGHT = 1800, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Game")

imagem = pygame.image.load('dinoSpritesheet.png')


# player infos
PLAYER_IMG = pygame.transform.scale(pygame.image.load('img_dinossauro.png'), (130, 120)).convert_alpha()
PLAYER_WIDTH, PLAYER_HEIGHT = 90, 80
PLAYER_VEL = 5

# obstacle infos
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 20, 50
cacto = imagem.subsurface((32*5, 0, 32, 32))
cacto = pygame.transform.scale(cacto, (OBSTACLE_WIDTH + 20, OBSTACLE_HEIGHT))

def draw(dino, obstaculos, score):
    WIN.fill('white')
    
    WIN.blit(PLAYER_IMG, (dino.x, dino.y))
    
    for obs in obstaculos:
        # pygame.draw.rect(WIN, 'green', (obs.x, obs.y, obs.width, obs.height))
        WIN.blit(cacto, (obs.x, obs.y))
    
    # pontuação
    font = pygame.font.SysFont('comicsans', 30)
    text = font.render('Pontuação: ' + str(score), 1, (0, 0, 0))
    WIN.blit(text, (10, 50))
    
    
    pygame.display.update()

class Dino(pygame.Rect):
    def __init__(self, x, y):
        super().__init__(x, y, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.vel = PLAYER_VEL
        self.img = PLAYER_IMG
        self.pulo = False
    
    def pular(self):
        if self.pulo == False:
            self.y -= 250
            self.pulo = True
    
    def gravidade(self):
            if self.y < 500:
                self.y += 5
            else:
                self.pulo = False
    
    def abaixar(self):
        self.y = 500

class Obstaculo(pygame.Rect):
    def __init__(self, x, y, velocidade):
        super().__init__(x, y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
        self.vel = velocidade

    
    def move(self):
        self.x -= self.vel       



def main():
    run = True
    dino = Dino(100, 500)
    clock = pygame.time.Clock()
    obstaculos = []
    obs_timer = 0
    max_obs = 7
    pontuacao = 0
    velocidade_obs = 5
    while run:
        clock.tick(60)
        obs_timer += 1
        pontuacao += round(1 * (velocidade_obs / 5), 0)
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

        dino.gravidade()
        
        # gerando obstaculos
        # aumentando a velocidade dos obstaculos
        if pontuacao % 1000 == 0:
            velocidade_obs += 3
        if len(obstaculos) == 0 or obstaculos[-1].x < random.randint(int(500 * (velocidade_obs / 3)), int(1000 * (velocidade_obs / 5))):
            if obs_timer > random.randint(100, 500) and len(obstaculos) <= max_obs and obs_timer % 10 == 0:
                obstaculos.append(Obstaculo(WIDTH, 550, velocidade_obs))
                obs_timer -= 100
        for obs in obstaculos:
            obs.move()
            if obs.colliderect(dino):
                run = False
                break
            if obs.x < -OBSTACLE_WIDTH:
                obstaculos.remove(obs)
        
        if run == False:
            break
            

        draw(dino, obstaculos, pontuacao)
            
    pygame.quit()

if __name__ == '__main__':
    main()