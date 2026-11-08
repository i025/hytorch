#include <stddef.h>
#include <stdlib.h>

#ifdef _WIN32
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT
#endif

/*
Row-major contract:
- A: (m, k)
- B: (k, n)
- C: (m, n)

Design notes:
- 4x4 micro-kernel does register-sized compute.
- Blocking improves cache locality.
- Packing converts strided source access into contiguous panel reads.
*/

#define MC 128
#define NC 128
#define KC 128

static int min_int(int a, int b) {
    return (a < b) ? a : b;
}

static int ceil_div4(int x) {
    return (x + 3) / 4;
}

/* ---------------- naive baseline (plain triple loop) ---------------- */

static int gemm_naive_core_f32(int m, int n, int k, const float *a, const float *b, float *c) {
    int i, j, p;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (p = 0; p < k; ++p) {
                sum += a[(size_t)i * (size_t)k + (size_t)p] * b[(size_t)p * (size_t)n + (size_t)j];
            }
            c[(size_t)i * (size_t)n + (size_t)j] = sum;
        }
    }
    return 0;
}

static int gemm_naive_core_f64(int m, int n, int k, const double *a, const double *b, double *c) {
    int i, j, p;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            double sum = 0.0;
            for (p = 0; p < k; ++p) {
                sum += a[(size_t)i * (size_t)k + (size_t)p] * b[(size_t)p * (size_t)n + (size_t)j];
            }
            c[(size_t)i * (size_t)n + (size_t)j] = sum;
        }
    }
    return 0;
}

/* ---------------- baseline (no packing) ---------------- */

static int gemm4x4_core_f32(int m, int n, int k, const float *a, const float *b, float *c) {
    int i, j, p;
    int m4 = (m / 4) * 4;
    int n4 = (n / 4) * 4;

    for (i = 0; i < m4; i += 4) {
        for (j = 0; j < n4; j += 4) {
            float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
            float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
            float c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
            float c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

            for (p = 0; p < k; ++p) {
                float a0 = a[(size_t)(i + 0) * (size_t)k + (size_t)p];
                float a1 = a[(size_t)(i + 1) * (size_t)k + (size_t)p];
                float a2 = a[(size_t)(i + 2) * (size_t)k + (size_t)p];
                float a3 = a[(size_t)(i + 3) * (size_t)k + (size_t)p];

                float b0 = b[(size_t)p * (size_t)n + (size_t)(j + 0)];
                float b1 = b[(size_t)p * (size_t)n + (size_t)(j + 1)];
                float b2 = b[(size_t)p * (size_t)n + (size_t)(j + 2)];
                float b3 = b[(size_t)p * (size_t)n + (size_t)(j + 3)];

                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }

            c[(size_t)(i + 0) * (size_t)n + (size_t)(j + 0)] = c00;
            c[(size_t)(i + 0) * (size_t)n + (size_t)(j + 1)] = c01;
            c[(size_t)(i + 0) * (size_t)n + (size_t)(j + 2)] = c02;
            c[(size_t)(i + 0) * (size_t)n + (size_t)(j + 3)] = c03;

            c[(size_t)(i + 1) * (size_t)n + (size_t)(j + 0)] = c10;
            c[(size_t)(i + 1) * (size_t)n + (size_t)(j + 1)] = c11;
            c[(size_t)(i + 1) * (size_t)n + (size_t)(j + 2)] = c12;
            c[(size_t)(i + 1) * (size_t)n + (size_t)(j + 3)] = c13;

            c[(size_t)(i + 2) * (size_t)n + (size_t)(j + 0)] = c20;
            c[(size_t)(i + 2) * (size_t)n + (size_t)(j + 1)] = c21;
            c[(size_t)(i + 2) * (size_t)n + (size_t)(j + 2)] = c22;
            c[(size_t)(i + 2) * (size_t)n + (size_t)(j + 3)] = c23;

            c[(size_t)(i + 3) * (size_t)n + (size_t)(j + 0)] = c30;
            c[(size_t)(i + 3) * (size_t)n + (size_t)(j + 1)] = c31;
            c[(size_t)(i + 3) * (size_t)n + (size_t)(j + 2)] = c32;
            c[(size_t)(i + 3) * (size_t)n + (size_t)(j + 3)] = c33;
        }
    }

    if (n4 < n) {
        for (i = 0; i < m4; ++i) {
            for (j = n4; j < n; ++j) {
                float sum = 0.0f;
                for (p = 0; p < k; ++p) {
                    sum += a[(size_t)i * (size_t)k + (size_t)p] * b[(size_t)p * (size_t)n + (size_t)j];
                }
                c[(size_t)i * (size_t)n + (size_t)j] = sum;
            }
        }
    }
    if (m4 < m) {
        for (i = m4; i < m; ++i) {
            for (j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (p = 0; p < k; ++p) {
                    sum += a[(size_t)i * (size_t)k + (size_t)p] * b[(size_t)p * (size_t)n + (size_t)j];
                }
                c[(size_t)i * (size_t)n + (size_t)j] = sum;
            }
        }
    }
    return 0;
}

static int gemm4x4_core_f64(int m, int n, int k, const double *a, const double *b, double *c) {
    int i, j, p;
    int m4 = (m / 4) * 4;
    int n4 = (n / 4) * 4;

    for (i = 0; i < m4; i += 4) {
        for (j = 0; j < n4; j += 4) {
            double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
            double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
            double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
            double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;

            for (p = 0; p < k; ++p) {
                double a0 = a[(size_t)(i + 0) * (size_t)k + (size_t)p];
                double a1 = a[(size_t)(i + 1) * (size_t)k + (size_t)p];
                double a2 = a[(size_t)(i + 2) * (size_t)k + (size_t)p];
                double a3 = a[(size_t)(i + 3) * (size_t)k + (size_t)p];

                double b0 = b[(size_t)p * (size_t)n + (size_t)(j + 0)];
                double b1 = b[(size_t)p * (size_t)n + (size_t)(j + 1)];
                double b2 = b[(size_t)p * (size_t)n + (size_t)(j + 2)];
                double b3 = b[(size_t)p * (size_t)n + (size_t)(j + 3)];

                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }

            c[(size_t)(i + 0) * (size_t)n + (size_t)(j + 0)] = c00;
            c[(size_t)(i + 0) * (size_t)n + (size_t)(j + 1)] = c01;
            c[(size_t)(i + 0) * (size_t)n + (size_t)(j + 2)] = c02;
            c[(size_t)(i + 0) * (size_t)n + (size_t)(j + 3)] = c03;

            c[(size_t)(i + 1) * (size_t)n + (size_t)(j + 0)] = c10;
            c[(size_t)(i + 1) * (size_t)n + (size_t)(j + 1)] = c11;
            c[(size_t)(i + 1) * (size_t)n + (size_t)(j + 2)] = c12;
            c[(size_t)(i + 1) * (size_t)n + (size_t)(j + 3)] = c13;

            c[(size_t)(i + 2) * (size_t)n + (size_t)(j + 0)] = c20;
            c[(size_t)(i + 2) * (size_t)n + (size_t)(j + 1)] = c21;
            c[(size_t)(i + 2) * (size_t)n + (size_t)(j + 2)] = c22;
            c[(size_t)(i + 2) * (size_t)n + (size_t)(j + 3)] = c23;

            c[(size_t)(i + 3) * (size_t)n + (size_t)(j + 0)] = c30;
            c[(size_t)(i + 3) * (size_t)n + (size_t)(j + 1)] = c31;
            c[(size_t)(i + 3) * (size_t)n + (size_t)(j + 2)] = c32;
            c[(size_t)(i + 3) * (size_t)n + (size_t)(j + 3)] = c33;
        }
    }

    if (n4 < n) {
        for (i = 0; i < m4; ++i) {
            for (j = n4; j < n; ++j) {
                double sum = 0.0;
                for (p = 0; p < k; ++p) {
                    sum += a[(size_t)i * (size_t)k + (size_t)p] * b[(size_t)p * (size_t)n + (size_t)j];
                }
                c[(size_t)i * (size_t)n + (size_t)j] = sum;
            }
        }
    }
    if (m4 < m) {
        for (i = m4; i < m; ++i) {
            for (j = 0; j < n; ++j) {
                double sum = 0.0;
                for (p = 0; p < k; ++p) {
                    sum += a[(size_t)i * (size_t)k + (size_t)p] * b[(size_t)p * (size_t)n + (size_t)j];
                }
                c[(size_t)i * (size_t)n + (size_t)j] = sum;
            }
        }
    }
    return 0;
}

/* ---------------- packed micro-kernel ---------------- */

/*
Packed A layout for panel (m_block x k_block):
- iterate row blocks of 4
- for each p in [0, k_block), store 4 rows A(i+0..i+3, p)
- if i+r >= m_block, store 0

So each 4-row block consumes k_block * 4 contiguous scalars.
*/
static void pack_a_panel_f32(int m_block, int k_block, const float *a, int lda, float *a_pack) {
    int i4, p, r;
    int num_i4 = ceil_div4(m_block);
    for (i4 = 0; i4 < num_i4; ++i4) {
        int row_base = i4 * 4;
        for (p = 0; p < k_block; ++p) {
            for (r = 0; r < 4; ++r) {
                int row = row_base + r;
                *a_pack++ = (row < m_block) ? a[(size_t)row * (size_t)lda + (size_t)p] : 0.0f;
            }
        }
    }
}

static void pack_a_panel_f64(int m_block, int k_block, const double *a, int lda, double *a_pack) {
    int i4, p, r;
    int num_i4 = ceil_div4(m_block);
    for (i4 = 0; i4 < num_i4; ++i4) {
        int row_base = i4 * 4;
        for (p = 0; p < k_block; ++p) {
            for (r = 0; r < 4; ++r) {
                int row = row_base + r;
                *a_pack++ = (row < m_block) ? a[(size_t)row * (size_t)lda + (size_t)p] : 0.0;
            }
        }
    }
}

/*
Packed B layout for panel (k_block x n_block):
- iterate col blocks of 4
- for each p in [0, k_block), store 4 cols B(p, j+0..j+3)
- if j+c >= n_block, store 0

So each 4-col block consumes k_block * 4 contiguous scalars.
*/
static void pack_b_panel_f32(int k_block, int n_block, const float *b, int ldb, float *b_pack) {
    int j4, p, c;
    int num_j4 = ceil_div4(n_block);
    for (j4 = 0; j4 < num_j4; ++j4) {
        int col_base = j4 * 4;
        for (p = 0; p < k_block; ++p) {
            for (c = 0; c < 4; ++c) {
                int col = col_base + c;
                *b_pack++ = (col < n_block) ? b[(size_t)p * (size_t)ldb + (size_t)col] : 0.0f;
            }
        }
    }
}

static void pack_b_panel_f64(int k_block, int n_block, const double *b, int ldb, double *b_pack) {
    int j4, p, c;
    int num_j4 = ceil_div4(n_block);
    for (j4 = 0; j4 < num_j4; ++j4) {
        int col_base = j4 * 4;
        for (p = 0; p < k_block; ++p) {
            for (c = 0; c < 4; ++c) {
                int col = col_base + c;
                *b_pack++ = (col < n_block) ? b[(size_t)p * (size_t)ldb + (size_t)col] : 0.0;
            }
        }
    }
}

static void gemm4x4_microkernel_f32(int k_block, const float *a_pack, const float *b_pack, float *c, int ldc) {
    int p;
    float c00 = c[0 * ldc + 0], c01 = c[0 * ldc + 1], c02 = c[0 * ldc + 2], c03 = c[0 * ldc + 3];
    float c10 = c[1 * ldc + 0], c11 = c[1 * ldc + 1], c12 = c[1 * ldc + 2], c13 = c[1 * ldc + 3];
    float c20 = c[2 * ldc + 0], c21 = c[2 * ldc + 1], c22 = c[2 * ldc + 2], c23 = c[2 * ldc + 3];
    float c30 = c[3 * ldc + 0], c31 = c[3 * ldc + 1], c32 = c[3 * ldc + 2], c33 = c[3 * ldc + 3];

    for (p = 0; p < k_block; ++p) {
        float a0 = a_pack[0], a1 = a_pack[1], a2 = a_pack[2], a3 = a_pack[3];
        float b0 = b_pack[0], b1 = b_pack[1], b2 = b_pack[2], b3 = b_pack[3];
        a_pack += 4;
        b_pack += 4;

        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
    }

    c[0 * ldc + 0] = c00; c[0 * ldc + 1] = c01; c[0 * ldc + 2] = c02; c[0 * ldc + 3] = c03;
    c[1 * ldc + 0] = c10; c[1 * ldc + 1] = c11; c[1 * ldc + 2] = c12; c[1 * ldc + 3] = c13;
    c[2 * ldc + 0] = c20; c[2 * ldc + 1] = c21; c[2 * ldc + 2] = c22; c[2 * ldc + 3] = c23;
    c[3 * ldc + 0] = c30; c[3 * ldc + 1] = c31; c[3 * ldc + 2] = c32; c[3 * ldc + 3] = c33;
}

static void gemm4x4_microkernel_f64(int k_block, const double *a_pack, const double *b_pack, double *c, int ldc) {
    int p;
    double c00 = c[0 * ldc + 0], c01 = c[0 * ldc + 1], c02 = c[0 * ldc + 2], c03 = c[0 * ldc + 3];
    double c10 = c[1 * ldc + 0], c11 = c[1 * ldc + 1], c12 = c[1 * ldc + 2], c13 = c[1 * ldc + 3];
    double c20 = c[2 * ldc + 0], c21 = c[2 * ldc + 1], c22 = c[2 * ldc + 2], c23 = c[2 * ldc + 3];
    double c30 = c[3 * ldc + 0], c31 = c[3 * ldc + 1], c32 = c[3 * ldc + 2], c33 = c[3 * ldc + 3];

    for (p = 0; p < k_block; ++p) {
        double a0 = a_pack[0], a1 = a_pack[1], a2 = a_pack[2], a3 = a_pack[3];
        double b0 = b_pack[0], b1 = b_pack[1], b2 = b_pack[2], b3 = b_pack[3];
        a_pack += 4;
        b_pack += 4;

        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
    }

    c[0 * ldc + 0] = c00; c[0 * ldc + 1] = c01; c[0 * ldc + 2] = c02; c[0 * ldc + 3] = c03;
    c[1 * ldc + 0] = c10; c[1 * ldc + 1] = c11; c[1 * ldc + 2] = c12; c[1 * ldc + 3] = c13;
    c[2 * ldc + 0] = c20; c[2 * ldc + 1] = c21; c[2 * ldc + 2] = c22; c[2 * ldc + 3] = c23;
    c[3 * ldc + 0] = c30; c[3 * ldc + 1] = c31; c[3 * ldc + 2] = c32; c[3 * ldc + 3] = c33;
}

/* ---------------- blocked + packed ---------------- */

static int gemm_blocked_packed_core_f32(int m, int n, int k, const float *a, const float *b, float *c) {
    int i, j, p;
    int jc, pc, ic;

    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            c[(size_t)i * (size_t)n + (size_t)j] = 0.0f;
        }
    }

    for (jc = 0; jc < n; jc += NC) {
        int jb = min_int(NC, n - jc);
        int jb4 = ceil_div4(jb);

        for (pc = 0; pc < k; pc += KC) {
            int pb = min_int(KC, k - pc);
            float *b_pack = (float *)malloc((size_t)pb * (size_t)jb4 * 4u * sizeof(float));
            if (b_pack == NULL) return -2;
            pack_b_panel_f32(pb, jb, b + (size_t)pc * (size_t)n + (size_t)jc, n, b_pack);

            for (ic = 0; ic < m; ic += MC) {
                int ib = min_int(MC, m - ic);
                int ib4 = ceil_div4(ib);
                int i_main = (ib / 4) * 4;
                int j_main = (jb / 4) * 4;
                float *a_pack = (float *)malloc((size_t)pb * (size_t)ib4 * 4u * sizeof(float));
                if (a_pack == NULL) {
                    free(b_pack);
                    return -2;
                }

                pack_a_panel_f32(ib, pb, a + (size_t)ic * (size_t)k + (size_t)pc, k, a_pack);

                for (i = 0; i < i_main; i += 4) {
                    const float *a_blk = a_pack + (size_t)(i / 4) * (size_t)pb * 4u;
                    for (j = 0; j < j_main; j += 4) {
                        const float *b_blk = b_pack + (size_t)(j / 4) * (size_t)pb * 4u;
                        float *c_blk = c + (size_t)(ic + i) * (size_t)n + (size_t)(jc + j);
                        gemm4x4_microkernel_f32(pb, a_blk, b_blk, c_blk, n);
                    }
                }

                /* Tail cleanup on borders that are not full 4x4 tiles. */
                for (i = 0; i < ib; ++i) {
                    int j_start = (i < i_main) ? j_main : 0;
                    for (j = j_start; j < jb; ++j) {
                        float sum = 0.0f;
                        const float *a_row = a + (size_t)(ic + i) * (size_t)k + (size_t)pc;
                        const float *b_col = b + (size_t)pc * (size_t)n + (size_t)(jc + j);
                        for (p = 0; p < pb; ++p) {
                            sum += a_row[p] * b_col[(size_t)p * (size_t)n];
                        }
                        c[(size_t)(ic + i) * (size_t)n + (size_t)(jc + j)] += sum;
                    }
                }

                free(a_pack);
            }
            free(b_pack);
        }
    }

    return 0;
}

static int gemm_blocked_packed_core_f64(int m, int n, int k, const double *a, const double *b, double *c) {
    int i, j, p;
    int jc, pc, ic;

    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            c[(size_t)i * (size_t)n + (size_t)j] = 0.0;
        }
    }

    for (jc = 0; jc < n; jc += NC) {
        int jb = min_int(NC, n - jc);
        int jb4 = ceil_div4(jb);

        for (pc = 0; pc < k; pc += KC) {
            int pb = min_int(KC, k - pc);
            double *b_pack = (double *)malloc((size_t)pb * (size_t)jb4 * 4u * sizeof(double));
            if (b_pack == NULL) return -2;
            pack_b_panel_f64(pb, jb, b + (size_t)pc * (size_t)n + (size_t)jc, n, b_pack);

            for (ic = 0; ic < m; ic += MC) {
                int ib = min_int(MC, m - ic);
                int ib4 = ceil_div4(ib);
                int i_main = (ib / 4) * 4;
                int j_main = (jb / 4) * 4;
                double *a_pack = (double *)malloc((size_t)pb * (size_t)ib4 * 4u * sizeof(double));
                if (a_pack == NULL) {
                    free(b_pack);
                    return -2;
                }

                pack_a_panel_f64(ib, pb, a + (size_t)ic * (size_t)k + (size_t)pc, k, a_pack);

                for (i = 0; i < i_main; i += 4) {
                    const double *a_blk = a_pack + (size_t)(i / 4) * (size_t)pb * 4u;
                    for (j = 0; j < j_main; j += 4) {
                        const double *b_blk = b_pack + (size_t)(j / 4) * (size_t)pb * 4u;
                        double *c_blk = c + (size_t)(ic + i) * (size_t)n + (size_t)(jc + j);
                        gemm4x4_microkernel_f64(pb, a_blk, b_blk, c_blk, n);
                    }
                }

                for (i = 0; i < ib; ++i) {
                    int j_start = (i < i_main) ? j_main : 0;
                    for (j = j_start; j < jb; ++j) {
                        double sum = 0.0;
                        const double *a_row = a + (size_t)(ic + i) * (size_t)k + (size_t)pc;
                        const double *b_col = b + (size_t)pc * (size_t)n + (size_t)(jc + j);
                        for (p = 0; p < pb; ++p) {
                            sum += a_row[p] * b_col[(size_t)p * (size_t)n];
                        }
                        c[(size_t)(ic + i) * (size_t)n + (size_t)(jc + j)] += sum;
                    }
                }

                free(a_pack);
            }
            free(b_pack);
        }
    }

    return 0;
}

/* ---------------- exported API ---------------- */

API_EXPORT int gemm4x4_f32(int m, int n, int k, const float *a, const float *b, float *c) {
    if (m < 0 || n < 0 || k < 0 || a == NULL || b == NULL || c == NULL) return -1;
    return gemm4x4_core_f32(m, n, k, a, b, c);
}

API_EXPORT int gemm4x4_f64(int m, int n, int k, const double *a, const double *b, double *c) {
    if (m < 0 || n < 0 || k < 0 || a == NULL || b == NULL || c == NULL) return -1;
    return gemm4x4_core_f64(m, n, k, a, b, c);
}

API_EXPORT int gemm_naive_f32(int m, int n, int k, const float *a, const float *b, float *c) {
    if (m < 0 || n < 0 || k < 0 || a == NULL || b == NULL || c == NULL) return -1;
    return gemm_naive_core_f32(m, n, k, a, b, c);
}

API_EXPORT int gemm_naive_f64(int m, int n, int k, const double *a, const double *b, double *c) {
    if (m < 0 || n < 0 || k < 0 || a == NULL || b == NULL || c == NULL) return -1;
    return gemm_naive_core_f64(m, n, k, a, b, c);
}

API_EXPORT int gemm_blocked_packed_f32(int m, int n, int k, const float *a, const float *b, float *c) {
    if (m < 0 || n < 0 || k < 0 || a == NULL || b == NULL || c == NULL) return -1;
    return gemm_blocked_packed_core_f32(m, n, k, a, b, c);
}

API_EXPORT int gemm_blocked_packed_f64(int m, int n, int k, const double *a, const double *b, double *c) {
    if (m < 0 || n < 0 || k < 0 || a == NULL || b == NULL || c == NULL) return -1;
    return gemm_blocked_packed_core_f64(m, n, k, a, b, c);
}
