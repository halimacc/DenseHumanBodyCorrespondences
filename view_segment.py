# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out
from config import *
import numpy as np
from matplotlib import cm
import sys
import pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random

# IMPORT OBJECT LOADER
from objloader import *


def load_segment(model_idx, mesh_idx, segment_idx):
    mesh_path = get_mesh_path(model_idx, mesh_idx)
    vertices, faces = load_mesh(mesh_path)

    segment_path = get_segment_path(model_idx, segment_idx)
    segment = np.loadtxt(segment_path)

    num_faces = faces.shape[0]

    random.seed()

    # create gl list1
    gl_list = glGenLists(1)
    glNewList(gl_list, GL_COMPILE)
    # glEnable(GL_TEXTURE_2D)
    # glFrontFace(GL_CCW)
    for i in xrange(num_faces):
        face = vertices[faces[i]]
        color = cm.hot(float(segment[i]) / num_segment)
        #color = cm.hot(float(i) / num_faces)

        #color = cm.hot(random.choice([0.8, 0.5]))
        glBegin(GL_TRIANGLES)
        for j in xrange(3):
            scale = 1
            glColor4fv(color)
            glVertex3f(face[j][0] * scale, face[j][1] * scale, face[j][2] * scale)
            # glVertex3fv(list(face[j]))
            
        glEnd()
    # glDisable(GL_TEXTURE_2D)
    glEndList()
    return gl_list


pygame.init()
viewport = (512, 512)
hx = viewport[0] / 2
hy = viewport[1] / 2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

glEnable(GL_DEPTH_TEST)
# most obj files expect to be smooth-shaded
glShadeModel(GL_SMOOTH)

model = load_segment(0, 0, 0)

clock = pygame.time.Clock()

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = viewport
gluPerspective(90.0, width / float(height), 1, 100.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

rx, ry = (0, 0)
tx, ty = (0, 0)
zpos = 5
rotate = move = False
while 1:
    clock.tick(30)
    for e in pygame.event.get():
        if e.type == QUIT:
            sys.exit()
        elif e.type == KEYDOWN and e.key == K_ESCAPE:
            sys.exit()
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == 4:
                zpos = max(1, zpos - 1)
            elif e.button == 5:
                zpos += 1
            elif e.button == 1:
                rotate = True
            elif e.button == 3:
                move = True
        elif e.type == MOUSEBUTTONUP:
            if e.button == 1:
                rotate = False
            elif e.button == 3:
                move = False
        elif e.type == MOUSEMOTION:
            i, j = e.rel
            if rotate:
                rx += i
                ry += j
            if move:
                tx += i
                ty -= j

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # RENDER OBJECT
    glTranslate(tx / 20., ty / 20., - zpos)
    glRotate(ry, 1, 0, 0)
    glRotate(rx, 0, 1, 0)
    # glCallList(obj.gl_list)
    glCallList(model)

    pygame.display.flip()
