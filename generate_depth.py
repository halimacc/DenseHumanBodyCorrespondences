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
from ctypes import *
import struct
import cv2


def get_gl_model(segment, vertices, faces):
    num_faces = faces.shape[0]

    # recenter model
    num_vertices = vertices.shape[0]
    center = np.array([0, 0, 0], dtype=np.float32)
    for i in xrange(num_vertices):
        for j in xrange(3):
            center[j] += vertices[i][j]
    center /= float(num_vertices)
    for i in xrange(num_vertices):
        for j in xrange(3):
            vertices[i][j] -= center[j]

    # create gl list1
    gl_list = glGenLists(1)
    glNewList(gl_list, GL_COMPILE)
    # glFrontFace(GL_CCW)
    for i in xrange(num_faces):
        face = vertices[faces[i]]
        color = int2color(segment[i])
        glBegin(GL_TRIANGLES)
        for j in xrange(3):
            glColor3fv(color)
            glVertex3fv(face[j].tolist())
        glEnd()
    glEndList()
    return gl_list


def draw_model(model, r, theta, d):

    glMatrixMode(GL_MODELVIEW)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # model 0
    glTranslate(0, 0, -2.5)
    glRotate(90, 0, 0, 1)
    glRotate(270, 1, 0, 0)

    glTranslate(0, 0, -d)
    glRotate(r, 1, 0, 0)
    glRotate(theta, 0, 0, 1)

    # model 1
    # glTranslate(0, 0, -2.5)
    # glRotate(90, 0, 1, 0)
    # glTranslate(0, 0, -d)
    # glRotate(r, 0, 1, 0)
    # glRotate(theta, 0, 0, 1)

    glCallList(model)

    # read pixels from buffer
    color_pixels = glReadPixels(0, 0, 512, 512, GL_RGB, GL_UNSIGNED_BYTE)
    depth_pixels = glReadPixels(0, 0, 512, 512, GL_DEPTH_COMPONENT, GL_FLOAT)

    return color_pixels, depth_pixels


if __name__ == '__main__':
    pygame.init()
    viewport = (512, 512)
    hx = viewport[0] / 2
    hy = viewport[1] / 2
    srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_FLAT)

    # set up view
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    width, height = viewport
    gluPerspective(70.0, width / float(height), znear, zfar)

    color_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, color_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

    depth_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 512, 512, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)

    fb = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fb)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0)

    for model_idx in xrange(0, 1):
        print "Generate depth images and label for model {0}...".format(model_idx)
        for segment_idx in xrange(69, 85):
            # load segment
            segment_path = get_segment_path(model_idx, segment_idx)
            segment = np.loadtxt(segment_path)
            print "Segment {0}...".format(segment_idx)
            for mesh_idx in xrange(num_mesh[model_idx]):

                if segment_idx == 69 and mesh_idx < 34:
                    continue

                print "Mesh {0}...".format(mesh_idx)
                # load mesh
                mesh_path = get_mesh_path(model_idx, mesh_idx)
                vertices, faces = load_mesh(mesh_path)

                if vertices == None or faces == None:
                    continue

                # get gl list
                gl_model = get_gl_model(segment, vertices, faces)

                exit
                # draw model from different perspective
                view_idx = -1
                for d in [0, -0.2]:
                    for theta in [0, -10, 10]:
                        for r in xrange(0, 360, 15):
                            view_idx += 1

                            color_path, depth_path = get_view_path(model_idx, segment_idx, mesh_idx, view_idx)
                            create_dir(color_path)
                            create_dir(depth_path)

                            color_pixels, depth_pixels = draw_model(gl_model, r, theta, d)

                            # save color
                            surface = pygame.image.fromstring(str(buffer(color_pixels)), (512, 512), 'RGB', True)
                            pygame.image.save(surface, color_path)

                            # save depth
                            zbuffer = np.array(struct.unpack('262144f', depth_pixels))
                            depth_gray_pixels = np.zeros(262144, dtype=np.uint8)
                            for i in xrange(262144):
                                depth = z2depth(zbuffer[i])
                                depth_gray_pixels[(511 - i / 512) * 512 + i % 512] = depth2gray(depth)
                            cv2.imwrite(depth_path, depth_gray_pixels.reshape(512, 512))

    glDeleteTextures(color_tex)
    glDeleteTextures(depth_tex)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDeleteFramebuffers([fb])
