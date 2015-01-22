from sklearn.neighbors import KNeighborsClassifier
from mnist import load_mnist
import numpy as np
import pygame, random

screen = pygame.display.set_mode((300,300))

draw_on = False
last_pos = (0, 0)
color = (255, 255, 255)
radius = 15

def roundline(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pygame.display.update(pygame.draw.circle(srf, color, (x, y), radius))

try:
    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT:
            raise StopIteration
        if e.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.circle(screen, color, e.pos, radius)
            draw_on = True
        if e.type == pygame.MOUSEBUTTONUP:
            draw_on = False
            drawnDigit = pygame.surfarray.array2d(pygame.transform.scale(screen,(28,28)))
        if e.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.display.update(pygame.draw.circle(screen, color, e.pos, radius))
                roundline(screen, color, e.pos, last_pos,  radius)
            last_pos = e.pos
        #pygame.display.flip()

except StopIteration:
    pass

pygame.quit()

drawnDigit = (np.transpose(drawnDigit)/16777215)
print drawnDigit
drawnDigit = drawnDigit.flatten()*255

X, y = load_mnist()
tX, ty = load_mnist(dataset="testing")

y = y[:,0]
ty = ty[:,0]
classif = KNeighborsClassifier(n_neighbors=10,weights='distance')
classif.fit(X,y)
print classif.predict(drawnDigit)
wrong = 0;
for i in range(0,len(ty)-1):
   if classif.predict(tX[i]) != ty[i]:
      wrong = wrong + 1
print "Test error: " + str((float(wrong)/(len(ty)-1)*100) + "%"

