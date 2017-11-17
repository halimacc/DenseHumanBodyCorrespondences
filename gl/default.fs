#version 330 core

flat in vec3 fColor[3];
in vec3 coord;
out vec4 outColor;

void main(){
    int i = (coord.x > coord.y && coord.x > coord.z) ? 0 : ((coord.y > coord.z) ? 1 : 2);
    outColor = vec4(fColor[i], 1.0);
}