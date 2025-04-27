#include "matrix_lib.h"
#include <stdlib.h>
#include <string.h>
#include <immintrin.h> 

/**
 * @brief multiplies tow matrices (using intel intrinsics lib)
 * 
 * @param m1 pointer to the first matrix
 * @param m2 pointer to the second matrix
 * @param r pointer to the result matrix
 * @return 0 in case of success or the error code
 */

int matrix_matrix_mult(matrix* m1, matrix* m2, matrix* r) {
    if(m1->cols != m2->rows) return 1;

    r->rows = m1->rows;
    r->cols = m2->cols;

    if(!m1 || !m2 || !r || !m1->values || m2->values|| !r->values) return -1;

    if(m1->rows % 8 != 0 || m1->cols % 8 != 0 || m2->rows % 8 != 0 || m2->cols % 8 != 0) return -3;

    memset(r->values, 0, sizeof(float) * r->rows * r->cols);

    float* m1_ptr = m1->values;
    float* m2_ptr = m2->values;
    float* r_ptr = r->values;

    for(int i = 0; i < m1->rows; i++) {
        float* m1_elem_ptr = m1_ptr;
        float* r_row_ptr = r_ptr;

        for(int j = 0; j < m1->cols; j++) {
            float value = *m1_elem_ptr;
            float* m2_row_ptr = m2_ptr + (j * m2->cols);

            float* r_col_ptr = r_row_ptr;
            __m256 value_vector = _mm256_set1_ps(value);

            for(int k = 0; k < m2->cols; k+=8) {
                __m256 m2_vals = _mm256_load_ps(m2_row_ptr);
                __m256 r_vals = _mm256_load_ps(r_col_ptr);
                __m256 mult = _mm256_mul_ps(m2_vals, value_vector);
                __m256 sum = _mm256_add_ps(r_vals, mult);
                _mm256_store_ps(r_col_ptr, sum);
                m2_row_ptr+=8;
                r_col_ptr+=8;
            }
            m1_elem_ptr++;
        }
        m1_ptr += m1->cols;
        r_ptr += r->cols;
    }

    return 0;
}


/**
 * @brief multiplies a matrix by a scalar value (using intel intrinsics lib)
 * 
 * @param scalar_value value that multiplies the matrix
 * @param m pointer to the original matrix
 * @param r pointer to the result matrix
 * @return 0 in case of success or the error code
 */
int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r){
    if(!m || !r || !m->values || !r->values) return -1;
    if (m->rows != r->rows || m->cols != r->cols) return -2;
    if(m->rows % 8 != 0 || m->cols % 8 != 0) return -3;

    __m256 scalarArr = _mm256_set1_ps(scalar_value); 
   
    int total = m->rows * m->cols;
    for (int i = 0; i < total; i += 8) {
        __m256 mLine = _mm256_load_ps(&m->values[i]);    
        __m256 resLine = _mm256_mul_ps(mLine, scalarArr);  
        _mm256_store_ps(&r->values[i], resLine);
    }
    return 0;    
}

/**
 * @brief multiplies a matrix by a scalar value (rows first)
 * 
 * @param scalar_value value that multiplies the matrix
 * @param m pointer to the original matrix
 * @param r pointer to the result matrix
 * @return 0 in case of success or the error code
 */

int scalar_matrix_mult_lines(float scalar_value, matrix* m, matrix* r) {
    if(!m || !r || !m->values || !r->values) return -1;
    if (m->rows != r->rows || m->cols != r->cols) return -2;

    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++) {
            int idx = i * m-> cols + j;

            r->values[idx] = scalar_value * m->values[idx];
        }
    }

    return 0;
}

/**
 * @brief multiplies a matrix by a scalar value (columns first)
 * 
 * @param scalar_value value that multiplies the matrix
 * @param m pointer to the original matrix
 * @param r pointer to the result matrix
 * @return 0 in case of success or the error code
 */

int scalar_matrix_mult_cols(float scalar_value, matrix *m, matrix *r) {
    if(!m || !r || !m->values || !r->values) return -1;
    if (m->rows != r->rows || m->cols != r->cols) return -2;


    for(int j = 0; j < m->cols; j++){
        for(int i = 0; i < m->rows; i++){
            int idx = i * m-> cols + j;

            r->values[idx] = scalar_value * m->values[idx];
        }
    }

    return 0;
}

/**
 * @brief multiplies a matrix by a scalar value (pointer-based version)
 * 
 * @param scalar_value value that multiplies the matrix
 * @param m pointer to the original matrix
 * @param r pointer to the result matrix
 * @return 0 in case of success or the error code
 */

int scalar_matrix_mult_ptr(float scalar_value, matrix* m, matrix* r) {
    if(!m || !r || !m->values || !r->values) return -1; 

    int size = m->rows * m->cols;

    float* m_ptr = m->values;
    float* r_ptr = r->values;
    
    for(int i = 0; i < size; i++) {
        *r_ptr = scalar_value * *(m_ptr);
        m_ptr++;
        r_ptr++;
    }

    return 0;
}


/**
 * @brief multiplies two matrices (traditional version)
 * 
 * @param m1 pointer to the first matrix
 * @param m2 pointer to the second matrix
 * @param r pointer to the result matrix
 * @return 0 in case of success or the error code
 */

int matrix_matrix_mult_trad(matrix *m1, matrix *m2, matrix *r){
    if(m1->cols != m2->rows) return 1;
    
    r->rows = m1->rows;
    r->cols = m2->cols;

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

/**
 * @brief multiplies two matrices (optimized version)
 * 
 * @param m1 pointer to the first matrix
 * @param m2 pointer to the second matrix
 * @param r pointer to the result matrix
 * @return 0 in case of success or the error code
 */

int matrix_matrix_mult_opt(matrix* m1, matrix* m2, matrix* r) {
    if(m1->cols != m2->rows) return 1;

    r->rows = m1->rows;
    r->cols = m2->cols;

    if(!r->values) return -1;

    memset(r->values, 0, sizeof(float) * r->rows * r->cols);

    float* m1_ptr = m1->values;
    float* m2_ptr = m2->values;
    float* r_ptr = r->values;

    for(int i = 0; i < m1->rows; i++) {
        float* m1_elem_ptr = m1_ptr;
        float* r_row_ptr = r_ptr;

        for(int j = 0; j < m1->cols; j++) {
            float value = *m1_elem_ptr;
            float* m2_row_ptr = m2_ptr + (j * m2->cols);

            float* r_col_ptr = r_row_ptr;

            for(int k = 0; k < m2->cols; k++) {
                *r_col_ptr += value * *m2_row_ptr;
                m2_row_ptr++;
                r_col_ptr++;
            }
            m1_elem_ptr++;
        }
        m1_ptr += m1->cols;
        r_ptr += r->cols;
    }


    return 0;
}
