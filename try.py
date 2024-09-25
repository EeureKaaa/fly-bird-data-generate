import os
import pygame
import sys

os.environ["SDL_AUDIODRIVER"] = "dummy"  # 禁用音频
pygame.init()
pygame.mixer.quit()  # 确保音频模块不启用
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption('Test Window')

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Quit event detected!")  # 用于调试的输出
            pygame.quit()
            sys.exit()
    
    print("Filling screen and updating display")  # 用于调试的输出
    screen.fill((0, 0, 0))
    pygame.display.flip()
