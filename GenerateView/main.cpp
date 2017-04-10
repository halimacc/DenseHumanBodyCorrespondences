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
//string get_segment_path(int model_idx, int segment_idx) {
//	return segment_dir + string_format("model_%02d_seg_%03d.seg", model_idx, segment_idx);
//}
//
//string get_mesh_path(int model_idx, int mesh_idx) {
//	return mesh_dir + string_format("%d\\mesh_%04d.obj", model_idx, mesh_idx);
//}
//
//string get_view_path(int model_idx, int segmentation_idx, int mesh_idx, int view_idx, string type) {
//	return view_dir + string_format("%d\\%03d\\%03d\\view_%03d_%s.png", model_idx, segmentation_idx, mesh_idx, view_idx, type.c_str());
//}
//
//string get_view_dir(int model_idx, int segmentation_idx, int mesh_idx) {
//	return view_dir + string_format("%d\\%03d\\%03d\\", model_idx, segmentation_idx, mesh_idx);
//}
//
//void load_segmentation(string seg_path, vector<int> &segmentation) {
//	segmentation.clear();
//	double belong;
//	ifstream inf(seg_path);
//	while (!inf.eof()) {
//		inf >> belong;
//		segmentation.push_back((int)belong);
//	}
//}
//
//vector<glm::vec3> int2color500;
//
//glm::vec3 int2color(int idx) {
//	glm::vec3 color;
//	idx += 1;
//	color.x = (idx / 100) / 5.0f;
//	color.y = ((idx % 100) / 10) / 10.0f;
//	color.z = (idx % 10) / 10.f;
//	return color;
//}
//
//void load_mesh(string mesh_path, vector<glm::vec3> &out_vertices, vector<glm::vec3> &out_colors, vector<int> segmentation) {
//	out_vertices.clear();
//	out_colors.clear();
//
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
//	for (int i = 0; i < segmentation.size(); ++i) {
//		glm::vec3 color = int2color500[segmentation[i]];
//		for (int j = 0; j < 3; ++j) {
//			out_vertices.push_back(tmp_vertices[indices[3 * i + j] - 1]);
//			out_colors.push_back(color);
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
//	// generate color texture
//	GLuint color_tex;
//	glGenTextures(1, &color_tex);
//	glBindTexture(GL_TEXTURE_2D, color_tex);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
//
//	// generate depth texture
//	GLuint depth_tex;
//	glGenTextures(1, &depth_tex);
//	glBindTexture(GL_TEXTURE_2D, depth_tex);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 512, 512, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
//
//	GLuint FramebufferName = 0;
//	glGenFramebuffers(1, &FramebufferName);
//	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0);
//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);
//
//	GLenum DrawBuffers[2] = { GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT};
//
//	// Create and compile our GLSL program from the shaders
//	GLuint programID = LoadShaders("VertexShader.vs", "FragmentShader.fs");
//
//	// Get a handle for our "MVP" uniform
//	GLuint MatrixID = glGetUniformLocation(programID, "MVP");
//	glm::mat4 Projection = glm::perspective(70.3f, 1.0f, znear, zfar);
//	glm::mat4 View = glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1), glm::vec3(0, -1, 0));
//
//	// prepare colors
//	for (int i = 0; i < 500; ++i) int2color500.push_back(int2color(i));
//
//	vector<int> segmentation;
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
//		cout << "Generate depth images and label for model " << model_idx << "...\n";
//
//		for (int segment_idx = 0; segment_idx < 100; ++segment_idx) {
//			cout << "Segment " << segment_idx << "...\n";
//
//			string segment_path = get_segment_path(model_idx, segment_idx);
//			load_segmentation(segment_path, segmentation);
//			for (int mesh_idx = 0; mesh_idx < num_mesh[model_idx]; ++mesh_idx) {
//				cout << "Mesh " << mesh_idx << "...\n";
//
//				string mesh_path = get_mesh_path(model_idx, mesh_idx);
//				load_mesh(mesh_path, vertices, colors, segmentation);
//
//				glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
//				glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);
//
//				glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
//				glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), &colors[0], GL_STATIC_DRAW);
//
//				int view_idx = -1;
//				float dis[] = {0, -0.2f};
//				float thetas[] = { 0, -10, 10 };
//				for (float d : dis) {
//					for (float theta : thetas) {
//						for (float r = 0; r < 360; r += 15) {
//							// draw model
//							// Clear the screen
//							glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//							// Use our shader
//							glUseProgram(programID);
//
//							// Model matrix : an identity matrix (model will be at the origin)
//							glm::mat4 Model = glm::mat4(1.0f);
//							Model = glm::translate(Model, glm::vec3(0, 0, -2.5 - d));
//							Model = glm::rotate(Model, glm::radians(90.0f + r), glm::vec3(0, 1, 0));
//							Model = glm::rotate(Model, glm::radians(theta), glm::vec3(0, 0, 1));
//
//							// Our ModelViewProjection : multiplication of our 3 matrices
//							glm::mat4 MVP = Projection * View * Model; // Remember, matrix multiplication is the other way around
//							glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
//
//							// 1rst attribute buffer : vertices
//							glEnableVertexAttribArray(0);
//							glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
//							glVertexAttribPointer(
//								0,                  // attribute
//								3,                  // size
//								GL_FLOAT,           // type
//								GL_FALSE,           // normalized?
//								0,                  // stride
//								(void*)0            // array buffer offset
//							);
//
//							// 2nd attribute buffer : UVs
//							glEnableVertexAttribArray(1);
//							glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
//							glVertexAttribPointer(
//								1,                                // attribute
//								3,                                // size
//								GL_FLOAT,                         // type
//								GL_FALSE,                         // normalized?
//								0,                                // stride
//								(void*)0                          // array buffer offset
//							);
//
//							glDrawArrays(GL_TRIANGLES, 0, vertices.size());
//
//							glDisableVertexAttribArray(0);
//							glDisableVertexAttribArray(1);
//
//							// Swap buffers
//							glfwSwapBuffers(window);
//
//							// get buffer
//							glReadPixels(0, 0, 512, 512, GL_RGB, GL_UNSIGNED_BYTE, color_pixels);
//							glReadPixels(0, 0, 512, 512, GL_DEPTH_COMPONENT, GL_FLOAT, z_pixels);
//							for (int i = 0; i < 512 * 512; ++i) {
//								depth_pixels[i] = z2gray(z_pixels[i]);
//							}
//							
//							view_idx++;
//							string color_path = get_view_path(model_idx, segment_idx, mesh_idx, view_idx, "color");
//							string depth_path = get_view_path(model_idx, segment_idx, mesh_idx, view_idx, "depth");
//							
//							//cv::imshow("test", depth);
//							cv::Mat depth_img(512, 512, CV_8U, depth_pixels);
//							cv::Mat color_img(512, 512, CV_8UC3, color_pixels);
//							cv::imwrite(color_path, color_img);
//							cv::imwrite(depth_path, depth_img);
//							//return 0;
//						}
//					}
//				}
//			}
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
