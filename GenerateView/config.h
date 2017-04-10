#pragma once

#include <string>
#include <glm/glm.hpp>
using namespace std;
using namespace glm;

template<typename ... Args>
string string_format(const std::string& format, Args ... args)
{
	size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	unique_ptr<char[]> buf(new char[size]);
	snprintf(buf.get(), size, format.c_str(), args ...);
	return string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

string project_dir = "D:\\Project\\DHBC\\";

int rgbRange = 256 * 256 * 256;
inline vec3 int2color(int integer, int max) {
	integer = integer * (rgbRange / max);
	vec3 color;
	color.x = (integer / 255 / 255) / 255.0; // r
	color.y = (integer / 255 % 255) / 255.0; // g
	color.z = (integer % 255) / 255.0; // b
	return color;
}

inline int color2int(int r, int g, int b, int max) {
	int integer = 255 * 255 * r + 255 * g + b;
	integer = integer / (rgbRange / max);
	return integer;
}