#version 120
uniform sampler2D dayTex;
uniform sampler2D nightTex;
uniform vec3 sunDir;
varying vec2 uv;
varying vec3 worldNormal;
void main() {
    float lit = dot(normalize(worldNormal), normalize(sunDir));
    float f = smoothstep(-0.1, 0.1, lit);          // 0 = night, 1 = day
    vec3 day = texture2D(dayTex, uv).rgb * clamp(lit, 0.05, 1.0);
    vec3 night = texture2D(nightTex, uv).rgb;      // city lights
    gl_FragColor = vec4(mix(night, day, f), 1.0);
}
