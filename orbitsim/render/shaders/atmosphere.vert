#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform vec3 wspos_view;        // camera world position (set from Python)
attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;
varying vec3 worldNormal;
varying vec3 viewDir;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vec3 wpos = (p3d_ModelMatrix * p3d_Vertex).xyz;
    worldNormal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
    viewDir = normalize(wspos_view - wpos);
}
