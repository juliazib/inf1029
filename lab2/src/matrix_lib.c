#include "matrix_lib.h"
#include <pthread.h>
#include <string.h>

typedef struct {
    float scalar;
    float* m_ptr;
    float* r_ptr;
    int start;
    int end;
} scalar_thread_args;

void* scalar_worker(void* arg) {
    scalar_thread_args* args = (scalar_thread_args*) arg;
    float* m = args->m_ptr + args->start;
    float* r = args->r_ptr + args->start;

    for (int i = args->start; i < args->end; i++) {
        *r = args->scalar * *m;
        r++;
        m++;
    }

    return NULL;
}

int scalar_matrix_mult(float scalar_value, matrix* m, matrix* r) {
    if (!m || !r || !m->values || !r->values) return -1;
    if (m->rows != r->rows || m->cols != r->cols) return -2;

    int num_threads = 8;
    int total = m->rows * m->cols;
    pthread_t threads[num_threads];
    scalar_thread_args args[num_threads];

    int chunk = total / num_threads;

    for (int i = 0; i < num_threads; i++) {
        args[i].scalar = scalar_value;
        args[i].m_ptr = m->values;
        args[i].r_ptr = r->values;
        args[i].start = i * chunk;
        args[i].end = (i == num_threads - 1) ? total : (i + 1) * chunk;
        pthread_create(&threads[i], NULL, scalar_worker, &args[i]);
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
} matrix_thread_args;

void* matrix_worker(void* arg) {
    matrix_thread_args* args = (matrix_thread_args*) arg;

    for (int i = args->start_row; i < args->end_row; i++) {
        float* row_r = args->r + i * args->m2_cols;
        float* row_m1 = args->m1 + i * args->m1_cols;

        for (int j = 0; j < args->m2_cols; j++) {
            float sum = 0;
            float* col_m2 = args->m2 + j;
            float* elem_m1 = row_m1;

            for (int k = 0; k < args->m1_cols; k++) {
                sum += (*elem_m1) * (*(col_m2));
                elem_m1++;
                col_m2 += args->m2_cols;
            }

            *row_r = sum;
            row_r++;
        }
    }

    return NULL;
}

int matrix_matrix_mult(matrix* m1, matrix* m2, matrix* r) {
    if (m1->cols != m2->rows) return 1;
    if (!m1 || !m2 || !r || !m1->values || !m2->values || !r->values) return -1;

    int num_threads = 8;

    r->rows = m1->rows;
    r->cols = m2->cols;

    memset(r->values, 0, sizeof(float) * r->rows * r->cols);

    pthread_t threads[num_threads];
    matrix_thread_args args[num_threads];

    int chunk = m1->rows / num_threads;

    for (int i = 0; i < num_threads; i++) {
        args[i].m1 = m1->values;
        args[i].m2 = m2->values;
        args[i].r = r->values;
        args[i].m1_rows = m1->rows;
        args[i].m1_cols = m1->cols;
        args[i].m2_cols = m2->cols;
        args[i].start_row = i * chunk;
        args[i].end_row = (i == num_threads - 1) ? m1->rows : (i + 1) * chunk;
        pthread_create(&threads[i], NULL, matrix_worker, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
