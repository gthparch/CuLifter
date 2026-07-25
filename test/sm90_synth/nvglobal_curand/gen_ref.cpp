#include <cstdio>
#include <curand.h>
int main() {
    const int N = 64;
    float buf[N];
    curandGenerator_t gen;
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, buf, N);
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(buf, 4, N, f);
    fclose(f);
    printf("%.9g %.9g %.9g\n", buf[0], buf[1], buf[63]);
    return 0;
}
