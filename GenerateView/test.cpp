//#pragma once
//
//// Include standard headers
//#include <stdio.h>
//#include <stdlib.h>
//#include <vector>
//#include <iostream>
//#include <string>
//#include <sstream>
//#include <memory>
//#include <fstream>
//#include <opencv2\opencv.hpp>
//using namespace std;
//
//// Include GLEW
//#include <GL\glew.h>
//
//// Include GLFW
//#include <GLFW\glfw3.h>
//GLFWwindow* window;
//
//// Include GLM
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//using namespace glm;
//
//#include "shader.hpp"
//
//string data_dir = "D:\\Project\\DHBC\\data\\";
//string segment_dir = data_dir + "segmentation\\";
//string mesh_dir = data_dir + "mesh\\";
//string view_dir = data_dir + "view\\";
//
//int num_mesh[] = { 72, 175, 150, 250, 250, 175, 175, 250, 250, 150, 175 };
//
//template<typename ... Args>
//string string_format(const std::string& format, Args ... args)
//{
//	size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
//	unique_ptr<char[]> buf(new char[size]);
//	snprintf(buf.get(), size, format.c_str(), args ...);
//	return string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
//}
//
//string get_mesh_path(int model_idx, int mesh_idx) {
//	return mesh_dir + string_format("%d\\mesh_%04d.obj", model_idx, mesh_idx);
//}
//
//
//
//void HSVtoRGB(float *r, float *g, float *b, float h, float s, float v)
//{
//	int i;
//	float f, p, q, t;
//	if (s == 0) {
//		// achromatic (grey)
//		*r = *g = *b = v;
//		return;
//	}
//	h /= 60;			// sector 0 to 5
//	i = floor(h);
//	f = h - i;			// factorial part of h
//	p = v * (1 - s);
//	q = v * (1 - s * f);
//	t = v * (1 - s * (1 - f));
//	switch (i) {
//	case 0:
//		*r = v;
//		*g = t;
//		*b = p;
//		break;
//	case 1:
//		*r = q;
//		*g = v;
//		*b = p;
//		break;
//	case 2:
//		*r = p;
//		*g = v;
//		*b = t;
//		break;
//	case 3:
//		*r = p;
//		*g = q;
//		*b = v;
//		break;
//	case 4:
//		*r = t;
//		*g = p;
//		*b = v;
//		break;
//	default:		// case 5:
//		*r = v;
//		*g = p;
//		*b = q;
//		break;
//	}
//}
//
//vector<glm::vec3> int2colorAny;
//
//glm::vec3 int2color(int idx) {
//	glm::vec3 color;
//
//	idx += 1;
//	float h = idx / 20000.0f * 360;
//
//	HSVtoRGB(&color.x, &color.y, &color.z, h, 1, 1);
//	//color.x = idx / 25 / 25 / 25.0f;
//	//color.y = idx / 25 % 25 / 25.0f;
//	//color.z = (idx % 25) / 25.0f;
//	return color;
//}
//
//void load_mesh(string mesh_path, vector<glm::vec3> &out_vertices, vector<glm::vec3> &out_colors) {
//	// load vertices and faces from file
//	vector<glm::vec3> tmp_vertices;
//	vector<int> indices;
//
//	glm::vec3 mean;
//	mean.x = mean.y = mean.z = 0;
//
//	ifstream fin(mesh_path);
//	while (!fin.eof()) {
//		char ch;
//		fin >> ch;
//		if (ch == 'v') {
//			glm::vec3 vertex;
//			fin >> vertex.x >> vertex.y >> vertex.z;
//			tmp_vertices.push_back(vertex);
//			mean += vertex;
//		}
//		else if (ch == 'f') {
//			int idx;
//			for (int i = 0; i < 3; ++i) {
//				fin >> idx;
//				indices.push_back(idx);
//			}
//		}
//	}
//
//	// recenter vertices
//	mean /= tmp_vertices.size();
//	for (int i = 0; i < tmp_vertices.size(); ++i)
//		tmp_vertices[i] -= mean;
//
//	// set vertices and colors
//	out_vertices.clear();
//	out_colors.clear();
//	for (int i = 0; i < indices.size() / 3; ++i) {
//		for (int j = 0; j < 3; ++j) {
//			out_vertices.push_back(tmp_vertices[indices[3 * i + j] - 1]);
//			out_colors.push_back(int2colorAny[indices[3 * i + j] - 1]);
//		}
//	}
//}
//
//float znear = 1.0f;
//float zfar = 3.55f;
//float b = zfar * znear / (znear - zfar);
//float a = -b / znear;
//inline unsigned char z2gray(float z) {
//	return 255 * (zfar - b / (z - a)) / (zfar - znear);
//
//}
//
//int main(void)
//{
//	// Initialise GLFW
//	if (!glfwInit())
//	{
//		fprintf(stderr, "Failed to initialize GLFW\n");
//		getchar();
//		return -1;
//	}
//
//	glfwWindowHint(GLFW_SAMPLES, 4);
//	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
//	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//
//	// Open a window and create its OpenGL context
//	window = glfwCreateWindow(512, 512, "Genrate Depth and Label Image", NULL, NULL);
//	if (window == NULL) {
//		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
//		getchar();
//		glfwTerminate();
//		return -1;
//	}
//	glfwMakeContextCurrent(window);
//
//	// Initialize GLEW
//	glewExperimental = true; // Needed for core profile
//	if (glewInit() != GLEW_OK) {
//		fprintf(stderr, "Failed to initialize GLEW\n");
//		getchar();
//		glfwTerminate();
//		return -1;
//	}
//
//	// black background
//	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
//
//	// Enable depth test
//	glEnable(GL_DEPTH_TEST);
//	// Accept fragment if it closer to the camera than the former one
//	glDepthFunc(GL_LESS);
//
//	// Cull triangles which normal is not towards the camera
//	glEnable(GL_CULL_FACE);
//
//	GLuint VertexArrayID;
//	glGenVertexArrays(1, &VertexArrayID);
//	glBindVertexArray(VertexArrayID);
//
//	GLuint vertexbuffer;
//	glGenBuffers(1, &vertexbuffer);
//
//	GLuint colorbuffer;
//	glGenBuffers(1, &colorbuffer);
//
//	// Create and compile our GLSL program from the shaders
//	GLuint programID = LoadShaders("VertexShader.vs", "FragmentShader.fs");
//
//	// Get a handle for our "MVP" uniform
//	GLuint MatrixID = glGetUniformLocation(programID, "MVP");
//	glm::mat4 Projection = glm::perspective(70.3f, 1.0f, znear, zfar);
//	glm::mat4 View = glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));
//
//	// prepare colors
//	for (int i = 0; i < 20000; ++i) {
//		
//		int2colorAny.push_back(int2color(i));
//	}
//
//	vector<glm::vec3> vertices;
//	vector<glm::vec3> colors;
//
//	cv::Mat depth;
//
//	unsigned char *color_pixels = new unsigned char[512 * 512 * 3];
//	float *z_pixels = new float[512 * 512];
//	unsigned char *depth_pixels = new unsigned char[512 * 512];
//	// get model
//	for (int model_idx = 1; model_idx < 10; ++model_idx) {
//		for (int mesh_idx = 0; mesh_idx < num_mesh[model_idx]; ++mesh_idx) {
//			string mesh_path = get_mesh_path(model_idx, mesh_idx);
//			load_mesh(mesh_path, vertices, colors);
//
//			glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
//			glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);
//
//			glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
//			glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_STATIC_DRAW);
//
//			// draw model
//			// Clear the screen
//			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//			// Use our shader
//			glUseProgram(programID);
//
//			// Model matrix : an identity matrix (model will be at the origin)
//			glm::mat4 Model = glm::mat4(1.0f);
//			Model = glm::translate(Model, glm::vec3(0, 0, -2.5));
//			Model = glm::rotate(Model, glm::radians(90.0f), glm::vec3(0, 1, 0));
//
//			// Our ModelViewProjection : multiplication of our 3 matrices
//			glm::mat4 MVP = Projection * View * Model; // Remember, matrix multiplication is the other way around
//			glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
//
//			// 1rst attribute buffer : vertices
//			glEnableVertexAttribArray(0);
//			glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
//			glVertexAttribPointer(
//				0,                  // attribute
//				3,                  // size
//				GL_FLOAT,           // type
//				GL_FALSE,           // normalized?
//				0,                  // stride
//				(void*)0            // array buffer offset
//			);
//
//			// 2nd attribute buffer : UVs
//			glEnableVertexAttribArray(1);
//			glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
//			glVertexAttribPointer(
//				1,                                // attribute
//				3,                                // size
//				GL_FLOAT,                         // type
//				GL_FALSE,                         // normalized?
//				0,                                // stride
//				(void*)0                          // array buffer offset
//			);
//
//			glDrawArrays(GL_TRIANGLES, 0, vertices.size());
//
//			glDisableVertexAttribArray(0);
//			glDisableVertexAttribArray(1);
//
//			// Swap buffers
//			glfwSwapBuffers(window);
//		}
//	}
//
//	glDeleteBuffers(1, &vertexbuffer);
//	glDeleteBuffers(1, &colorbuffer);
//	glDeleteProgram(programID);
//	glDeleteVertexArrays(1, &VertexArrayID);
//
//	// Close OpenGL window and terminate GLFW
//	glfwTerminate();
//
//	return 0;
//}
