/*
fire :
https://www.shadertoy.com/view/XsXXRN

paper : 
https://www.shadertoy.com/view/MtXSWr

*/

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;




#define T u_time
#define r(v,t) { float a = (t)*T, c=cos(a),s=sin(a); v*=mat2(c,s,-s,c); }
#define SQRT3_2  1.26
#define SQRT2_3  1.732
#define smin(a,b) (1./(1./(a)+1./(b)))

// --- noise functions from https://www.shadertoy.com/view/XslGRr
// Created by inigo quilez - iq/2013
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

const mat3 m = mat3( 0.00,  0.80,  0.60,
           		    -0.80,  0.36, -0.48,
             		-0.60, -0.48,  0.64 );

float hash( float n ) {
    return fract(sin(n)*43758.5453);
}

float noise( in vec3 x ) { // in [0,1]
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.-2.*f);

    float n = p.x + p.y*57. + 113.*p.z;

    float res = mix(mix(mix( hash(n+  0.), hash(n+  1.),f.x),
                        mix( hash(n+ 57.), hash(n+ 58.),f.x),f.y),
                    mix(mix( hash(n+113.), hash(n+114.),f.x),
                        mix( hash(n+170.), hash(n+171.),f.x),f.y),f.z);
    return res;
}

float fbm( vec3 p ) { // in [0,1]
    float f;
    f  = 0.5000*noise( p ); p = m*p*2.02;
    f += 0.2500*noise( p ); p = m*p*2.03;
    f += 0.1250*noise( p ); p = m*p*2.01;
    f += 0.0625*noise( p );
    return f;
}
// --- End of: Created by inigo quilez --------------------

// --- more noise

#define snoise(x) (2.*noise(x)-1.)

float sfbm( vec3 p ) { // in [-1,1]
    float f;
    f  = 0.5000*snoise( p ); p = m*p*2.02;
    f += 0.2500*snoise( p ); p = m*p*2.03;
    f += 0.1250*snoise( p ); p = m*p*2.01;
    f += 0.0625*snoise( p );
    return f;
}

#define sfbm3(p) vec3(sfbm(p), sfbm(p-327.67), sfbm(p+327.67))

// --- using the base ray-marcher of Trisomie21: https://www.shadertoy.com/view/4tfGRB#

vec4 bg = vec4(0,0,.2,0);//change to flame color


// fire
float rand(vec2 n) {
    return fract(cos(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(vec2 n) {
    const vec2 d = vec2(0.0, 1.0);
    vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
    return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

float fbm(vec2 n) {
    float total = 0.0, amplitude = 1.0;
    for (int i = 0; i < 4; i++) {
        total += noise(n) * amplitude;
        n += n;
        amplitude *= 0.5;
    }
    return total;
}

vec4 flame(){
    const vec3 c1 = vec3(0.5, 0.0, 0.1);
    const vec3 c2 = vec3(0.9, 0.0, 0.0);
    const vec3 c3 = vec3(0.2, 0.0, 0.0);
    const vec3 c4 = vec3(1.0, 0.9, 0.0);
    const vec3 c5 = vec3(0.1);
    const vec3 c6 = vec3(0.9);
    
    vec2 speed = vec2(0.7, 0.4);
    float shift = 1.6;
    float alpha = 1.0;

    vec2 p = gl_FragCoord.xy * 8.0 / u_resolution.xx;
    float q = fbm(p - u_time * 0.1);
    vec2 r = vec2(fbm(p + q + u_time * speed.x - p.x - p.y), fbm(p + q - u_time * speed.y));
    vec3 c = mix(c1, c2, fbm(p + r)) + mix(c3, c4, r.x) - mix(c5, c6, r.y);
    vec4 fragColor = vec4(c * cos(shift * gl_FragCoord.y / u_resolution.y), alpha);
    
    return fragColor*.5;
}


void main() {
    vec2 w = gl_FragCoord.xy;
    vec4 f;//fragcolor

    vec4 p = vec4(w,0,1)/u_resolution.yyxy-.5, d,c; p.x-=.4; // init ray 
    (p.xz,.013); r(p.yz,.02); r(p.xy,.1);   // camera rotations
    d = p;                                 // ray dir = ray0-vec3(0)
    p = -vec4(0,.5,1,0)*T*.1;

    //float closest = 999.0;

    float x1,x2,x=1e2;
    
    for (float i=1.; i>0.; i-=.01)  {
        
        // vec4 u = floor(p/8.), t = mod(p, 8.)-4., ta; // objects id + local frame
        vec4 u = floor(p/vec4(5,5,1,1)),
            t = p, ta,v;
        
        // r(t.xy,u.x); r(t.xz,u.y); r(t.yz,1.);    // objects rotations
        u = sin(78.*(u+u.yzxw));                    // randomize ids
        // t -= u;                                  // jitter positions
        c = p/p*2.;
		t.xyz += sfbm3(t.xyz/2.+vec3(-.1*T,0,0));
 
		x1 = abs(mod(t.z,.5)-.5/2.); 
        // x1 = length(t.xyz)-4.; x = max(x, x1);
        // max(ta.x,max(ta.y,ta.z))
        x2 = abs(mod(t.x-.5,1.)-.5)-.4; x = max(-x2,x1);
        if (x2<.1) c = mix(c, vec4(0.4784, 0.4706, 0.4706, 1.0), clamp(cos(T/10.),0.,1.));
       
        if(x<.001) // hit !
            { f = mix(flame(),c,i*i); break;  } 
        
        p += d*x;          
     }
        
    gl_FragColor = f;
}


