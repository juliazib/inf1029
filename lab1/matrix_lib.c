#include "matrix_lib.h"

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
    // if(m1->cols != m2->rows) return 1;
    
    // r->rows = m1->cols;
    // r->cols = m2->rows;
    
    // float soma = 0;
    // int idx = 0;
    // for(int j = 0; j < r->cols; j++){
        
    //     for(int i = 0; i < r->rows; i++){
    //         float tmp = *(m1->values + i * j + i) *(m2->values + j * i + i)/    
        
    //     }    
    // }
    
    return 0;   

}

