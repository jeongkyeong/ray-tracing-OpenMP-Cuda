#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define CUDA 0
#define OPENMP 1
#define SPHERES 20

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 2e10f
#define DIM 2048


typedef struct sphere{
	float   r, b, g;
	float   radius;
	float   x, y, z;

}Sphere;

float hit(float, float, float*, Sphere);
void kernel(int, int, Sphere*, unsigned char*);


float hit(float ox, float oy, float *n, Sphere s) {
	float dx = ox - s.x;
	float dy = oy - s.y;
	float radius = s.radius;
	if (dx*dx + dy*dy < radius*radius) {
		float dz = sqrtf(radius*radius - dx*dx - dy*dy);
		*n = dz / sqrtf(radius * radius);
		return dz + s.z;
	}
	return -INF;
}

void kernel(int x, int y, Sphere* s, unsigned char* ptr)
{
	int offset = x + y*DIM;
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);
	int i = 0;
	
	//printf("x:%d, y:%d, ox:%f, oy:%f\n",x,y,ox,oy);

	float r = 0, g = 0, b = 0;
	float   maxz = -INF;

#pragma omp parallel for schedule(dynamic)
	for (i = 0; i<SPHERES; i++) {
		float   n;
		float   t = hit(ox, oy, &n, s[i]);
		if (t > maxz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	}

	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;

	
}

void ppm_write(unsigned char* bitmap, int xdim, int ydim, FILE* fp)
{
	int i, x, y;
	fprintf(fp, "P3\n");
	fprintf(fp, "%d %d\n", xdim, ydim);
	fprintf(fp, "255\n");

	for (y = 0; y<ydim; y++) {
		for (x = 0; x<xdim; x++) {
			i = x + y*xdim;
			fprintf(fp, "%d %d %d ", bitmap[4 * i], bitmap[4 * i + 1], bitmap[4 * i + 2]);
		}
		fprintf(fp, "\n");
	}
}

int main(int argc, char* argv[])
{
		
	int no_threads;
	int option;
	int x, y;
	int i = 0;
	unsigned char* bitmap;
	double exetime = 0;

	srand(time(NULL));

	if (argc != 3) {
		printf("> a.out [option] [filename.ppm]\n");
		printf("[option] 0: CUDA, 1~16: OpenMP using 1~16 threads\n");
		printf("for example, '> a.out 8 result.ppm' means executing OpenMP with 8 threads\n");
		exit(0);
	}
	FILE* fp = fopen(argv[2], "w");

	if (strcmp(argv[1], "0") == 0) option = CUDA;
	else {
		option = OPENMP;
		no_threads = atoi(argv[1]);
	}

	omp_set_num_threads(no_threads);

	exetime = omp_get_wtime();

	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);

	for (i = 0; i < SPHERES; i++) {
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(2000.0f) - 1000;
		temp_s[i].y = rnd(2000.0f) - 1000;
		temp_s[i].z = rnd(2000.0f) - 1000;
		temp_s[i].radius = rnd(200.0f) + 40;
		//printf("%d: r: %f, g: %f, b: %f\n", i, temp_s[i].r, temp_s[i].g, temp_s[i].b);
	}

	bitmap = (unsigned char*)malloc(sizeof(unsigned char)*DIM*DIM * 4);

#pragma omp parallel for schedule(dynamic) default(shared) private(x,y,i)
	for (x = 0; x < DIM; x++){
		for (y = 0; y < DIM; y++){
			kernel(x, y, temp_s, bitmap);
		}
	}
	
	
	exetime = omp_get_wtime() - exetime;
	printf("OpenMP ray tracing (# thread: %d)\n", no_threads);
	printf("Elapse time: %lf\n", exetime);
	ppm_write(bitmap, DIM, DIM, fp);
	printf("[%s] was generated.\n", argv[2]);
	fclose(fp);
	free(bitmap);
	free(temp_s);

	return 0;
}
