// References: 
// 【多盞燈光】 https://www.shadertoy.com/view/Xls3R7 
// (參考他多盞燈光的處理方式，不過我這邊加入struct概念，讓物件更容易管理及使用。)
// 【場景製作】 https://www.youtube.com/watch?v=PGtv-dBi2wE

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;



vec3 normalMap(vec3 p, vec3 n);
float noise_3(in vec3 p); //亂數範圍 [0,1]
float smin( float a, float b, float k );



float sdSphere(vec3 p){//sphere scene
    vec4 sphere1 = vec4(50.,-70.,1.,28.); 
    vec4 sphere2 = vec4(8.,20.,2.,15.); 
    vec4 sphere3 = vec4(70.,20.,10.,10.); 
    //前後 / 左右 / 顏色

    float dSphere1 = distance(p, sphere1.xyz)-sphere1.w;
    float dSphere2 = distance(p, sphere2.xyz)-sphere2.w;
    float dSphere3 = distance(p, sphere3.xyz)-sphere3.w;

    return min(min(dSphere1, dSphere2),min(min(dSphere1, dSphere2),dSphere3));
}

// SDF of Plane
float plasma(vec3 r) {
	float mx = r.x + u_time / 2.;
	mx += 60.0 * sin((r.y + mx) / 40.0 + u_time / 30.);
	float my = r.y + u_time / 5.;
	my += 80.0 * cos(r.x / 20.0 + u_time / 6.);
	return r.z - (sin(mx / 15.0) + sin(my / 5.0) + 5.);
}

float map(in vec3 p)
{
    float bump = noise_3(p)*.4;
    vec3 p1 = p + bump;

    bump = noise_3(p*.2)*5.;
    vec3 p2 = p + bump;

    //return min(plasma(p1), sdSphere(p2));
    //return mix(plasma(p1), sdSphere(p2),.9);

    return smin(plasma(p1), sdSphere(p2), 5.);//將石頭融入波紋
}

float RayMarching(vec3 ro,vec3 rd){
    float dO = 0.;

    for(int i = 0 ; i < 64 ; i++){
        vec3 p = ro+rd*dO;
        float ds = map(p);
        dO += ds;
        if(ds<0.01 || dO>100.)
            break;
    }
    
    return dO;
}

const float coeiff = 0.25;
const vec3 totalSkyLight = vec3(0.3, 0.5, 1.0);

vec3 mie(float dist, vec3 sunL){
    return max(exp(-pow(dist, 0.25)) * sunL - 0.4, 0.0);
}

vec3 getSky(vec2 uv){
	
	vec2 sunPos = vec2(0.5, cos(u_time * 0.5 + 3.14 * 0.564));
    
    float sunDistance = distance(uv, clamp(sunPos, -1.0, 1.0));
	
	float scatterMult = clamp(sunDistance, 0.0, 1.0);
	float sun = clamp(1.0 - smoothstep(0.01, 0.011, scatterMult), 0.0, 1.0);
	
	float dist = uv.y;
	dist = (coeiff * mix(scatterMult, 1.0, dist)) / dist;
    
    vec3 mieScatter = mie(sunDistance, vec3(1.0));
	
	vec3 color = dist * totalSkyLight;
    
    color = max(color, 0.0);
    
	color = max(mix(pow(color, 1.0 - color),
        color / (2.0 * color + 0.5 - color),
        clamp(sunPos.y * 2.0, 0.0, 1.0)),0.0)
        + sun + mieScatter;
	
	color *=  (pow(1.0 - scatterMult, 10.0) * 10.0) + 1.0;
	
	float underscatter = distance(sunPos.y * 0.5 + 0.5, 1.0);
	
	color = mix(color, vec3(0.0), clamp(underscatter, 0.0, 1.0));
	
	return color;	
}

//=== sky ===
float fbm(in vec2 uv);
vec3 getSkyFBM(vec3 e) {	//二維雲霧
	vec3 f=e;
	float m = 2.0 * sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
	vec2 st= vec2(-f.x/m + .5, -f.y/m + .5);
	//vec3 ret=texture2D(iChannel0, st).xyz;
	float fog= fbm(0.6*st+vec2(-0.2*u_time, -0.02*u_time))*0.5+0.3;
    return vec3(fog);
}

vec3 sky_color(vec3 e) {	//漸層藍天空色 //藍色波長衰減快
    e.y = max(e.y,0.0);
    vec3 ret;
    ret.x = pow(1.0-e.y,3.0);
    ret.y = pow(1.0-e.y, 1.2);
    ret.z = 0.8+(1.0-e.y)*0.3;    
    return ret;
}

vec3 getSkyALL(vec3 e)
{	
	return sky_color(e);
}

vec3 gradient( in vec3 p ) //尚未normalize
{
	const float d = 0.001;
	vec3 grad = vec3(map(p+vec3(d,0,0))-map(p-vec3(d,0,0)),
                     map(p+vec3(0,d,0))-map(p-vec3(0,d,0)),
                     map(p+vec3(0,0,d))-map(p-vec3(0,0,d)));
	return grad;
}





void main() {
    //vec2 st = (gl_FragCoord.xy/u_resolution.xy)*2.-1.;
    float c, s;
	float vfov = 3.14159 / 2.3;

	vec3 cam = vec3(0.0, 0.0, 35.0);

    vec2 fragCoord = gl_FragCoord.xy;
	vec2 uv = (fragCoord.xy / u_resolution.xy) - 0.5;
	uv.x *= u_resolution.x / u_resolution.y;
	uv.y *= -1.0;

	vec3 dir = vec3(0.0, 0.0, -1.0);

	float xrot = vfov * length(uv);

    c = cos(xrot);
	s = sin(xrot);
	dir = mat3(1.0, 0.0, 0.0,
	           0.0,   c,  -s,
	           0.0,   s,   c) * dir;

	c = normalize(uv).x;
	s = normalize(uv).y;
	dir = mat3(  c,  -s, 0.0,
	             s,   c, 0.0,
	           0.0, 0.0, 1.0) * dir;

	c = cos(0.7);
	s = sin(0.7);
	dir = mat3(  c, 0.0,   s,
	           0.0, 1.0, 0.0,
	            -s, 0.0,   c) * dir;

    // ro -> ray origin
    vec3 ro = vec3(0.0, -.2, 0.0);

    // rd -> ray direction
    vec3 rd = normalize(vec3(uv.xy,1.));

    // t -> distance
    //float t = RayMarching(ro,rd);
    float dist = RayMarching(cam, dir);
	
    // p -> vertex pos
    //vec3 p = ro+rd*t;
    vec3 pos = cam + dist * dir;
    
    // n -> normal dir
    vec3 n=normalize(gradient(pos));
    //vec3 bump=normalMap(p*1.0,n);
    //n+=bump*0.05;

    vec3 color = getSky(fragCoord.xy / u_resolution.x);
	color = color / (2.0 * color + 4. - color);//skycolor



    vec4 fragColor=vec4(1.);
	fragColor.rgb = mix(
        //color,// sun movement
        //vec3(11,52,110)/255.,//pool
        vec3(55,60,56)/255.,
        //getSkyALL(reflect(rd,n)),//lava

        mix(
			vec3(0.0, 0.0, 0.0),
			vec3(1.0, 1.0, 1.0),
			pos.z / 15.0 //water reflection
		),
		1.0 / (dist / 25.0) // ink painting
        //1.0 / (dist / 120.0)// lava
        //1.0 / (dist / 10.0)   // pool
	);

    
    gl_FragColor = fragColor;
}








//=== 2d noise functions ===
vec2 hash2( vec2 x )			//亂數範圍 [-1,1]
{
    const vec2 k = vec2( 0.3183099, 0.3678794 );
    x = x*k + k.yx;
    return -1.0 + 2.0*fract( 16.0 * k*fract( x.x*x.y*(x.x+x.y)) );
}
float gnoise( in vec2 p )		//亂數範圍 [-1,1]
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
    vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash2( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     	    dot( hash2( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                	     mix( dot( hash2( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     	    dot( hash2( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}
float fbm(in vec2 uv)		//亂數範圍 [-1,1]
{
	float f;				//fbm - fractal noise (4 octaves)
	mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
	f   = 0.5000*gnoise( uv ); uv = m*uv;		  
	f += 0.2500*gnoise( uv ); uv = m*uv;
	f += 0.1250*gnoise( uv ); uv = m*uv;
	f += 0.0625*gnoise( uv ); uv = m*uv;
	return f;
}



//=== 3d noise functions ===
float hash11(float p) {
    return fract(sin(p * 727.1)*43758.5453123);
}
float hash12(vec2 p) {
	float h = dot(p,vec2(127.1,311.7));	
    return fract(sin(h)*43758.5453123);
}
vec3 hash31(float p) {
	vec3 h = vec3(1275.231,4461.7,7182.423) * p;	
    return fract(sin(h)*43758.543123);
}

// 3d noise
float noise_3(in vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);	
	vec3 u = f*f*(3.0-2.0*f);
    
    vec2 ii = i.xy + i.z * vec2(5.0);
    float a = hash12( ii + vec2(0.0,0.0) );
	float b = hash12( ii + vec2(1.0,0.0) );    
    float c = hash12( ii + vec2(0.0,1.0) );
	float d = hash12( ii + vec2(1.0,1.0) ); 
    float v1 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
    
    ii += vec2(5.0);
    a = hash12( ii + vec2(0.0,0.0) );
	b = hash12( ii + vec2(1.0,0.0) );    
    c = hash12( ii + vec2(0.0,1.0) );
	d = hash12( ii + vec2(1.0,1.0) );
    float v2 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
        
    return max(mix(v1,v2,u.z),0.0);
}


//=== 3d noise functions p/n ===
vec3 smoothSampling2(vec2 uv)
{
    const float T_RES = 32.0;
    return vec3(gnoise(uv*T_RES)); //讀取亂數函式
}

float triplanarSampling(vec3 p, vec3 n)
{
    float fTotal = abs(n.x)+abs(n.y)+abs(n.z);
    return  (abs(n.x)*smoothSampling2(p.yz).x
            +abs(n.y)*smoothSampling2(p.xz).x
            +abs(n.z)*smoothSampling2(p.xy).x)/fTotal;
}

const mat2 m2 = mat2(0.90,0.44,-0.44,0.90);
float triplanarNoise(vec3 p, vec3 n)
{
    const float BUMP_MAP_UV_SCALE = 0.2;
    float fTotal = abs(n.x)+abs(n.y)+abs(n.z);
    float f1 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    p.xy = m2*p.xy;
    p.xz = m2*p.xz;
    p *= 2.1;
    float f2 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    p.yx = m2*p.yx;
    p.yz = m2*p.yz;
    p *= 2.3;
    float f3 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    return f1+0.5*f2+0.25*f3;
}

vec3 normalMap(vec3 p, vec3 n)
{
    float d = 0.005;
    float po = triplanarNoise(p,n);
    float px = triplanarNoise(p+vec3(d,0,0),n);
    float py = triplanarNoise(p+vec3(0,d,0),n);
    float pz = triplanarNoise(p+vec3(0,0,d),n);
    return normalize(vec3((px-po)/d,
                          (py-po)/d,
                          (pz-po)/d));
}

float smin( float a, float b, float k )
{
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*k*(1.0/4.0);
}

