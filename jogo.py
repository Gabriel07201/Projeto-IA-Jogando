import pygame
import random

pygame.font.init()

# Window size
WIDTH, HEIGHT = 1000, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Game")

# player infos
PLAYER_IMG = pygame.transform.scale(pygame.image.load('img_dinossauro.png'), (100, 100)).convert_alpha()
PLAYER_WIDTH, PLAYER_HEIGHT = 100, 100
PLAYER_VEL = 5

# obstacle infos
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 30, 100
OBSTACLE_VEL = 5

def draw(dino, obstaculos):
    WIN.fill('white')
    
    WIN.blit(PLAYER_IMG, (dino.x, dino.y))
    
    for obs in obstaculos:
        pygame.draw.rect(WIN, 'green', (obs.x, obs.y, obs.width, obs.height))
    
    
    pygame.display.update()

class Dino():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.vel = PLAYER_VEL
        self.img = PLAYER_IMG
        self.pulo = False
    
    def pular(self):
        if self.pulo == False:
            self.y -= 200
            self.pulo = True
    
    def gravidade(self):
            if self.y < 500:
                self.y += 10
            else:
                self.pulo = False

class Obstaculo():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = OBSTACLE_WIDTH
        self.height = OBSTACLE_HEIGHT
        self.vel = OBSTACLE_VEL
    
    def move(self):
        self.x -= self.vel

                  


def main():
    run = True
    dino = Dino(100, 500)
    clock = pygame.time.Clock()
    obstaculos = []
    obs_timer = 0
    while run:
        clock.tick(60)
        obs_timer += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        
        # Pulo
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            dino.pular()

        dino.gravidade()
        
        # gerando obstaculos
        # if obs_timer > 20:
        if random.randint(0, 10) == 1:
            obstaculos.append(Obstaculo(WIDTH, 500))
            obs_timer -= 10
        for obs in obstaculos:
            obs.move()

        draw(dino, obstaculos)
            
    pygame.quit()

if __name__ == '__main__':
    main()