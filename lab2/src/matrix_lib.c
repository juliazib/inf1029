#include "matrix_lib.h"
#include <pthread.h>
#include <string.h>

typedef struct {
    float scalar;
    float* m_start;
    float* r_start;
    int length;
} scalar_thread_args_ptr;

void* scalar_worker_ptr(void* arg) {
    scalar_thread_args_ptr* args = (scalar_thread_args_ptr*) arg;
    float* m_ptr = args->m_start;
    float* r_ptr = args->r_start;
    int len = args->length;

    for (int i = 0; i < len; i++) {
        *r_ptr = args->scalar * *m_ptr;
        r_ptr++;
        m_ptr++;
    }

    return NULL;
}

int scalar_matrix_mult(float scalar_value, matrix* m, matrix* r) {
    if (!m || !r || !m->values || !r->values) return -1;
    if (m->rows != r->rows || m->cols != r->cols) return -2;

    int num_threads = 8;
    int total = m->rows * m->cols;

    pthread_t threads[num_threads];
    scalar_thread_args_ptr args[num_threads];

    int chunk = total / num_threads;
    int remainder = total % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int current_chunk = chunk + (i < remainder ? 1 : 0);  

        args[i].scalar = scalar_value;
        args[i].m_start = m->values + i * chunk + (i < remainder ? i : remainder);
        args[i].r_start = r->values + i * chunk + (i < remainder ? i : remainder);
        args[i].length = current_chunk;

        pthread_create(&threads[i], NULL, scalar_worker_ptr, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}


typedef struct {
    float* m1;
    float* m2;
    float* r;
    int m1_rows;
    int m1_cols;
    int m2_cols;
    int start_row;
    int end_row;
} thread_args;

void* optimized_worker(void* arg) {
    thread_args* args = (thread_args*) arg;

    float* m1_ptr = args->m1 + args->start_row * args->m1_cols;
    float* r_ptr  = args->r  + args->start_row * args->m2_cols;
    float* m2_ptr = args->m2;

    for (int i = args->start_row; i < args->end_row; i++) {
        float* m1_elem_ptr = m1_ptr;
        float* r_row_ptr   = r_ptr;

        for (int j = 0; j < args->m1_cols; j++) {
            float value = *m1_elem_ptr;
            float* m2_row_ptr = m2_ptr + j * args->m2_cols;
            float* r_col_ptr  = r_row_ptr;

            for (int k = 0; k < args->m2_cols; k++) {
                *r_col_ptr += value * *m2_row_ptr;
                r_col_ptr++;
                m2_row_ptr++;
            }
            m1_elem_ptr++;
        }

        m1_ptr += args->m1_cols;
        r_ptr  += args->m2_cols;
    }

    return NULL;
}

int matrix_matrix_mult(matrix* m1, matrix* m2, matrix* r) {
    if (!m1 || !m2 || !r || !m1->values || !m2->values || !r->values) return -1;
    if (m1->cols != m2->rows) return 1;

    int num_threads = 8;

    r->rows = m1->rows;
    r->cols = m2->cols;

    memset(r->values, 0, sizeof(float) * r->rows * r->cols);

    pthread_t threads[num_threads];
    thread_args args[num_threads];

    int chunk = m1->rows / num_threads;

    for (int i = 0; i < num_threads; i++) {
        args[i].m1 = m1->values;
        args[i].m2 = m2->values;
        args[i].r  = r->values;
        args[i].m1_rows = m1->rows;
        args[i].m1_cols = m1->cols;
        args[i].m2_cols = m2->cols;
        args[i].start_row = i * chunk;
        args[i].end_row = (i == num_threads - 1) ? m1->rows : (i + 1) * chunk;
        pthread_create(&threads[i], NULL, optimized_worker, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}