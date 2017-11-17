from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import numpy as np

def LoadProgram(shaderPathList):
    shaderTypeMapping = {
        'vs': GL_VERTEX_SHADER,
        'gs': GL_GEOMETRY_SHADER,
        'fs': GL_FRAGMENT_SHADER
        }
    shaderTypeList = [shaderTypeMapping[shaderType] for shaderPath in shaderPathList for shaderType in shaderTypeMapping if shaderPath.endswith(shaderType)]
    shaders = []
    for i in range(len(shaderPathList)):
        shaderPath = shaderPathList[i]
        shaderType = shaderTypeList[i]

        with open(shaderPath, 'r') as f:
            shaderData = f.read()
        shader = glCreateShader(shaderType)
        glShaderSource(shader, shaderData)
        glCompileShader(shader)
        shaders.append(shader)

    program = glCreateProgram()
    for shader in shaders:
        glAttachShader(program, shader)
    glLinkProgram(program)
    for shader in shaders:
        glDetachShader(program, shader)
        glDeleteShader(shader)
    return program

class GLRenderer(object):
    def __init__(self, name, size, pos, toTexture=False, shaderPathList=None):
        self.width, self.height = size
        self.toTexture = toTexture

        glutInit()
        displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL | GLUT_BORDERLESS | GLUT_CAPTIONLESS
        glutInitDisplayMode (displayMode)
        glutInitWindowPosition(pos[0], pos[1])
        glutInitWindowSize(self.width, self.height)
        self.window = glutCreateWindow(name)
        #glutFullScreen()
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        self.vertexArr = glGenVertexArrays(1)
        glBindVertexArray(self.vertexArr)
        self.vertexBuf = glGenBuffers(1)
        self.colorBuf = glGenBuffers(1)
        glClearColor(0.0, 0.0, 0.0, 0.0)

        self.toTexture = toTexture
        if toTexture:
            self.texColor = glGenTextures(1);
            glBindTexture(GL_TEXTURE_2D, self.texColor)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

            self.texDepth = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texDepth)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.width, self.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)

            self.frameBuf = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuf)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texColor, 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.texDepth, 0)

        if not shaderPathList:
            shaderPathList = [os.path.join('gl', sh) for sh in ['default.vs', 'default.gs', 'default.fs']]
        #shaderPathList = [os.path.join('glhelper', sh) for sh in ['default.vs']]
        self.program = LoadProgram(shaderPathList)
        self.mvpMatrix = glGetUniformLocation(self.program, 'MVP')

    def draw(self, vertices, colors, mvp):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)
        glUniformMatrix4fv(self.mvpMatrix, 1, GL_FALSE, mvp)

        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuf)
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(
            0,                  # attribute
            vertices.shape[1],                  # size
            GL_FLOAT,           # type
            GL_FALSE,           # normalized?
            0,                  # stride
            None            # array buffer offset
        );

        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.colorBuf)
        glBufferData(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW);
        glVertexAttribPointer(
            1,                                # attribute
            colors.shape[1],                                # size
            GL_FLOAT,                         # type
            GL_FALSE,                         # normalized?
            0,                                # stride
            None                          # array buffer offset
        );

        glDrawArrays(GL_TRIANGLES, 0, vertices.shape[0])
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glUseProgram(0)
        glutSwapBuffers()

        rgb = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, outputType=None)
        z = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT, outputType=None)
        rgb = np.flip(np.flip(rgb, 0), 1)
        z = np.flip(np.flip(z, 0), 1)
        return rgb, z
