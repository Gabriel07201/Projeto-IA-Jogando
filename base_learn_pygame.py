import pygame
import time
import random

pygame.font.init()

# Window size
WIDTH, HEIGHT = 1000, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino block Game")

# player infos
PLAYER_IMG = pygame.transform.scale(pygame.image.load('img_dinossauro.png'), (100, 100)).convert_alpha()
PLAYER_WIDTH, PLAYER_HEIGHT = 100, 100
GRAVITY = 2

# obstacle infos
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 30, 100
OBSTACLE_VEL = 5


# draw function
def draw(player, obstacle):
    WIN.fill('white')
    
    WIN.blit(PLAYER_IMG, (player.x, player.y))
    
    for obs in obstacle:
        pygame.draw.rect(WIN, 'green', obs)
        
    pygame.display.update()


def main():
    run = True
    
    # pos_x, pos_y, width, height
    player = pygame.Rect(100, 500, PLAYER_WIDTH, PLAYER_HEIGHT)
    clock = pygame.time.Clock()
    
    obstacle_add_increment = 2000
    obstacle_count = 0
    obstacles = []
    hit = False
    
    while run:
        obstacle_count += clock.tick(60)
        
        if obstacle_count > obstacle_add_increment:
            obstacle_count = 0
            obstacle = pygame.Rect(WIDTH, 500, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
            obstacles.append(obstacle)
            
            obstacle_add_increment = max(200, obstacle_add_increment - 100)
            obstacle_count = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and player.y == 500:
            player.y -= 150
        if player.y < 500:
            player.y += GRAVITY
        
        
        
        for obs in obstacles[:]:
            obs.x -= OBSTACLE_VEL
            if obs.colliderect(player):
                hit = True
                break
        
        if hit:
            break
        
        draw(player, obstacles)
        
    pygame.quit()


if __name__ == "__main__":
    main()