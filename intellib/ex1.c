#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>


float* multiplicaPorEscalar(int tamanho,int escalar, float* v){
    __m256 p = _mm256_set_ps(escalar, escalar, escalar, escalar, escalar, escalar, escalar, escalar);
    __m256 k; 
    
    //faltou o alligned aloc no vetor de destino;
    float* v2 =  (float*)_mm_malloc(sizeof(float)*tamanho, 32);
    int it = tamanho / 8;
    for(int i = 0; i < it; i++){
        // __m256 q = _mm256_set_ps(v[8 * i], v[8 * i + 1], v[8 * i + 2], v[8 * i + 3], v[8 * i + 4], v[8 * i + 5], v[8 * i + 6], v[8 * i + 7]);
        // ^ essa instrucao nao é necessaria, pode se usar load_ps pra facilitar;
        
        __m256 q = _mm256_load_ps(&v[8*i]);
        k = _mm256_mul_ps(q, p);
        
        // for(int j = 0; j < 8; j++){
        //     v2[i * 8 + j] = k[j];
        // }
        //mesma coisa aqui, pode ser substituida por _mm256_store_ps;

        _mm256_store_ps(&v2[i*8], k);

    }

    return v2;
}

float* preencheVetorSize(float* v, int size){
    //faltou o alligned aloc no vetor de tamanho variavel

    float* l = (float*)_mm_malloc(size * sizeof(float), 32);    
    for(int i = 0; i<size; i++) {
        l[i] = v[i % 8];
    }
    return l;
}



int main(void) {

    float v[8] = {3.2f, 4.5f, 6.7f, 1.03f, 8.8f, 9.87f, 5.5f, 6.4f};

    float* mult10 = preencheVetorSize(v, 80);
    float* res = multiplicaPorEscalar(80, 1,mult10);
    
    for(int i = 0; i < 80; i++) {
        printf("%f\n", res[i]);
    }


 
    return 0;
}
