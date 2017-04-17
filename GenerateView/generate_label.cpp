#include "config.h"
#include "shader.h"
#include <GLFW\glfw3.h>
#include <GL\glew.h>
#include <glm\gtc\matrix_transform.hpp>
#include <opencv2\opencv.hpp>
#include <Windows.h>
using namespace std;
using namespace cv;
using namespace glm;

GLFWwindow* window;
GLuint ProgramID;
GLuint MatrixID;
GLuint VertexArrayID;
GLuint VertexBuffer;
GLuint ColorBuffer;

void initial_opengl() {
	// Initialise GLFW
	glfwInit();
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(512, 512, "Genrate Depth and Label Image", NULL, NULL);
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	glewInit();

	// black background
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_CULL_FACE);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	// Generate Buffers
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);
	glGenBuffers(1, &VertexBuffer);
	glGenBuffers(1, &ColorBuffer);

	// Create and compile our GLSL program from the shaders
	ProgramID = LoadShaders("VertexShader.vs", "GeometryShader.gs", "FragmentShader.fs");
	MatrixID = glGetUniformLocation(ProgramID, "MVP");
}

void draw_model(vector<vec3> &vertices, vector<vec3> &colors, mat4 MVP) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(ProgramID);

	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(
		0,                  // attribute
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	// 2nd attribute buffer : UVs
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, ColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_STATIC_DRAW);
	glVertexAttribPointer(
		1,                                // attribute
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);

	glDrawArrays(GL_TRIANGLES, 0, vertices.size());

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	// Swap buffers
	glfwSwapBuffers(window);
}

int main(void)
{
	initial_opengl();

	unsigned char *color_pixels = new unsigned char[512 * 512 * 3];
	float *z_pixels = new float[512 * 512];
	unsigned char *depth_pixels = new unsigned char[512 * 512];

	// generate color texture
	GLuint color_tex;
	glGenTextures(1, &color_tex);
	glBindTexture(GL_TEXTURE_2D, color_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

	// generate depth texture
	GLuint depth_tex;
	glGenTextures(1, &depth_tex);
	glBindTexture(GL_TEXTURE_2D, depth_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 512, 512, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

	GLuint FramebufferName = 0;
	glGenFramebuffers(1, &FramebufferName);
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);

	GLenum DrawBuffers[2] = { GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT };

	glm::mat4 Projection = glm::perspective(70.3f, 1.0f, znear, zfar);
	glm::mat4 View = glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1), glm::vec3(0, -1, 0));

	cout << string_format("Prepare dense and sparse segmentation color...");
	vector<vec3> segmentation_color_dense;
	for (int i = 0; i < 500; ++i) {
		segmentation_color_dense.push_back(int2color(i + 1, 500 + 1));
	}
	vector<vec3> segmentation_color_sparse;
	for (int i = 0; i < 32; ++i) {
		segmentation_color_sparse.push_back(int2color(i + 1, 32 + 1));
	}

	for (int model_idx = 0; model_idx < 1; ++model_idx) {
		cout << string_format("Generate label and depth data for model %d...\n", model_idx);

		vector<vec3> vertices;
		vector<uint32> faces;
		string example_mesh_path = get_mesh_path(model_idx, 0);
		load_mesh(example_mesh_path, vertices, faces);
		int vertex_cnt = vertices.size();
		int face_cnt = faces.size() / 3;
		cout << string_format("Model %d has %d vertices and %d faces.", model_idx, vertex_cnt, face_cnt);
		cout << "Prepare vertex colors...\n";
		vector<vec3> vertex_color;
		for (int i = 0; i < vertex_cnt; ++i) {
			vertex_color.push_back(int2color(i + 1, vertex_cnt + 1));
		}

		for (int mesh_idx = 0; mesh_idx < num_mesh[model_idx]; ++mesh_idx) {
			cout << string_format("Model %d, Mesh %d...\n", model_idx, mesh_idx);
			string mesh_path = get_mesh_path(model_idx, mesh_idx);
			load_mesh(mesh_path, vertices, faces);

			// prepare vertex buffer and vertex color buffer
			vector<vec3> vertex_buffer;
			vector<vec3> vertex_color_buffer;
			for (int i = 0; i < faces.size(); ++i) {
				vertex_buffer.push_back(vertices[faces[i]]);
				vertex_color_buffer.push_back(vertex_color[faces[i]]);
			}

			for (int segmentation_idx = 0; segmentation_idx < 100; ++segmentation_idx) {
				cout << string_format("Model %d, Mesh %d, MeshSegmentation %d...\n", model_idx, mesh_idx, segmentation_idx);
				vector<uint> segmentation;
				string segmentation_path = get_segmentation_path(model_idx, segmentation_idx);
				load_segmentation(segmentation_path, segmentation);

				vector<vec3> face_color_buffer;
				for (int i = 0; i < faces.size(); ++i) {
					face_color_buffer.push_back(segmentation_color_dense[segmentation[i / 3]]);
				}

				// draw model from 144 different perspective
				int view_idx = -1;
				float dis[] = { 0, 0.2f };
				float thetas[] = { 0, -5, 5 };
				for (float d : dis) {
					for (float theta : thetas) {
						for (float r = 0; r < 360; r += 15) {
							view_idx++;

							glm::mat4 Model = glm::mat4(1.0f);
							Model = glm::translate(Model, glm::vec3(0, 0, -2.5 - d));
							if (model_idx == 0) {
								Model = glm::rotate(Model, glm::radians(90.0f), glm::vec3(0, 0, 1));
								Model = glm::rotate(Model, glm::radians(270.0f + r), glm::vec3(1, 0, 0));
								Model = glm::rotate(Model, glm::radians(theta), glm::vec3(0, 0, 1));
							}
							else {
								Model = glm::rotate(Model, glm::radians(90.0f + r), glm::vec3(0, 1, 0));
								Model = glm::rotate(Model, glm::radians(theta), glm::vec3(0, 0, 1));
							}

							// Our ModelViewProjection : multiplication of our 3 matrices
							glm::mat4 MVP = Projection * View * Model; // Remember, matrix multiplication is the other way around

							// draw model
							draw_model(vertex_buffer, face_color_buffer, MVP);

							glReadPixels(0, 0, 512, 512, GL_RGB, GL_UNSIGNED_BYTE, color_pixels);
							string segmentation_view_path = get_segmentation_view_path(model_idx, mesh_idx, segmentation_idx, view_idx);
							Mat simg(512, 512, CV_8UC3, color_pixels);
							imwrite(segmentation_view_path, simg);

							if (segmentation_idx == 0) {
								draw_model(vertex_buffer, vertex_color_buffer, MVP);

								glReadPixels(0, 0, 512, 512, GL_RGB, GL_UNSIGNED_BYTE, color_pixels);
								string vertex_view_path = get_vertex_view_path(model_idx, mesh_idx, view_idx);
								Mat vimg(512, 512, CV_8UC3, color_pixels);
								imwrite(vertex_view_path, vimg);

								glReadPixels(0, 0, 512, 512, GL_DEPTH_COMPONENT, GL_FLOAT, z_pixels);
								for (int i = 0; i < 512 * 512; ++i) {
									depth_pixels[i] = z2gray(z_pixels[i]);
								}
								string depth_view_path = get_depth_view_path(model_idx, mesh_idx, view_idx);
								Mat dimg(512, 512, CV_8U, depth_pixels);
								imwrite(depth_view_path, dimg);
							}
						}
					}
				}

			}
		}
	}

	glDeleteTextures(1, &color_tex);
	glDeleteTextures(1, &depth_tex);
	glDeleteFramebuffers(1, &FramebufferName);
	glDeleteProgram(ProgramID);
	glDeleteBuffers(1, &VertexBuffer);
	glDeleteBuffers(1, &ColorBuffer);
	glDeleteVertexArrays(1, &VertexArrayID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}