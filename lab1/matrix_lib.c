#include "matrix_lib.h"
#include <stdlib.h>

int scalar_matrix_mult_rows(float scalar_value, matrix *m, matrix *r){
    if(!m || !r || !m->values || !r->values) return -1;


    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++) {
            int idx = i * m-> cols + j;

            r->values[idx] = scalar_value * m->values[idx];
        }
    }

    return 0;   
}

int scalar_matrix_mult_cols(float scalar_value, matrix *m, matrix *r) {
    if(!m || !r || !m->values || !r->values) return -1;

    for(int j = 0; j < m->cols; j++){
        for(int i = 0; i < m->rows; i++){
            int idx = i * m-> cols + j;

            r->values[idx] = scalar_value * m->values[idx];
        }
    }

    return 0;
}

int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r){
    if(m1->cols != m2->rows) return 1;
    
    r->rows = m1->rows;
    r->cols = m2->cols;

    r->values = (float*) malloc(r->rows * r->cols * sizeof(float));

    if(!r->values) return -1;

    for(int i1 = 0; i1 < m1->rows; i1++) {
        for(int j2 = 0; j2 < m2->cols; j2++) {
            r->values[i1 * r->cols + j2] = 0;

            for(int j1 = 0; j1 < m1->cols; j1++) {
                r->values[i1 * r->cols + j2] += m1->values[i1 * m1->cols + j1] * m2->values[j1 * m2->cols + j2];
            }
        }
    }
    
    return 0;   

}

