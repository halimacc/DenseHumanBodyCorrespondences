#pragma once

#include "tinyply.h"
#include <string>
#include <vector>
#include <fstream>
#include <glm/glm.hpp>

using std::string;
using glm::vec3;
using tinyply::PlyFile;
using std::vector;
using std::ifstream;
using glm::uint;

template<typename ... Args>
string string_format(const std::string& format, Args ... args)
{
	size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	unique_ptr<char[]> buf(new char[size]);
	snprintf(buf.get(), size, format.c_str(), args ...);
	return string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

string project_dir = "D:\\Project\\DHBC\\";
string data_dir = project_dir + "data\\";
string segment_dir = data_dir + "segmentation\\";
string mesh_dir = data_dir + "mesh\\";
string view_dir = data_dir + "view\\";
int num_mesh[] = { 71, 175, 150, 250, 250, 175, 175, 250, 250, 150, 175 };

string get_segmentation_path(int model_idx, int segment_idx) {
	return segment_dir + string_format("model_%02d_seg_%03d.seg", model_idx, segment_idx);
}

string get_mesh_path(int model_idx, int mesh_idx) {
	if (model_idx == 0)
		return mesh_dir + string_format("%d\\mesh%03d.ply", model_idx, mesh_idx);
	return mesh_dir + string_format("%d\\mesh_%04d.obj", model_idx, mesh_idx);
}

string get_segmentation_view_path(int model_idx, int mesh_idx, int segmentation_idx, int view_idx) {
	return view_dir + string_format("model_%02d\\mesh_%03d\\segmentation_%03d\\%03d.png", model_idx, mesh_idx, segmentation_idx, view_idx);
}

string get_vertex_view_path(int model_idx, int mesh_idx, int view_idx) {
	return view_dir + string_format("model_%02d\\mesh_%03d\\vertex\\%03d.png", model_idx, mesh_idx, view_idx);
}

string get_depth_view_path(int model_idx, int mesh_idx, int view_idx) {
	return view_dir + string_format("model_%02d\\mesh_%03d\\depth\\%03d.png", model_idx, mesh_idx, view_idx);
}

void load_segmentation(string seg_path, vector<uint> &segmentation) {
	segmentation.clear();
	double belong;
	ifstream inf(seg_path);
	while (!inf.eof()) {
		inf >> belong;
		segmentation.push_back((uint)belong);
	}
}

int rgbRange = 256 * 256 * 256;
inline vec3 int2color(int integer, int max = rgbRange) {
	integer = integer * (rgbRange / max);
	vec3 color;
	color.x = (integer / 256 / 256) / 255.0; // r
	color.y = (integer / 256 % 256) / 255.0; // g
	color.z = (integer % 256) / 255.0; // b
	return color;
}

inline int color2int(int r, int g, int b, int max = rgbRange) {
	int integer = 255 * 255 * r + 255 * g + b;
	integer = integer / (rgbRange / max);
	return integer;
}

void load_mesh(string mesh_path, vector<vec3> &out_vertices, vector<uint> &out_faces, bool binary = false) {
	out_vertices.clear();
	out_faces.clear();
	
	ifstream fin(mesh_path, binary ? std::ios::binary : std::ios::in);

	if (mesh_path[mesh_path.size() - 1] == 'j') { // obj file
		while (!fin.eof()) {
			char ch;
			fin >> ch;
			if (ch == 'v') {
				vec3 vertex;
				fin >> vertex.x >> vertex.y >> vertex.z;
				out_vertices.push_back(vertex);
			}
			else if (ch == 'f') {
				uint idx;
				for (int i = 0; i < 3; ++i) {
					fin >> idx;
					out_faces.push_back(idx - 1);
				}
			}
		}
	}

	if (mesh_path[mesh_path.size() - 1] == 'y') { // ply file
		PlyFile file(fin);
		vector<float> vertices;
		int vertex_cnt = file.request_properties_from_element("vertex", { "x", "y", "z" }, vertices);
		int face_cnt = file.request_properties_from_element("face", { "vertex_indices" }, out_faces, 3);
		file.read(fin);
		for (int i = 0; i < vertex_cnt; ++i) out_vertices.push_back(vec3(vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]));
	}

	// recenter model
	vec3 mean(0, 0, 0);
	for (int i = 0; i < out_vertices.size(); ++i)
		mean += out_vertices[i];
	mean /= out_vertices.size();
	for (int i = 0; i < out_vertices.size(); ++i)
		out_vertices[i] -= mean;
}

float znear = 1.0f;
float zfar = 3.55f;
float b = zfar * znear / (znear - zfar);
float a = -b / znear;
inline unsigned char z2gray(float z) {
	return 255 * (zfar - b / (z - a)) / (zfar - znear);
}
