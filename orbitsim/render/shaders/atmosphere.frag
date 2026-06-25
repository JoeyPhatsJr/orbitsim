#version 120
varying vec3 worldNormal;
varying vec3 viewDir;
void main() {
    float rim = 1.0 - max(dot(normalize(worldNormal), normalize(viewDir)), 0.0);
    float a = pow(rim, 3.0);                        // bright at the limb
    gl_FragColor = vec4(0.3, 0.6, 1.0, a * 0.9);    // sky-blue halo
}
