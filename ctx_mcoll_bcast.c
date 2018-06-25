#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include <shmem.h>

#include <cycles.c>

#define SIZE 32*1024*1024
#define NUM_THREADS 40

double cpu_mhz;
shmem_ctx_h ctx[40];

unsigned long long shmem_wtime()
{
    return get_cycles();
}

/* This is the shmem_broadcast64 (linear) code */
void my_bcast64 (void * target, const void * source, size_t nelems,
                 int PE_root, int PE_start, int logPE_stride, int PE_size,
                 long * pSync)
{
    const int typed_nelems = nelems * 8;
    const int step = 1 << logPE_stride;
    const int root = (PE_root * step) + PE_start;
    const int me = shmem_my_pe();//GET_STATE(mype);
    shmem_barrier (PE_start, logPE_stride, PE_size, pSync);
    if (me != root) {
        shmem_getmem(target, source, typed_nelems, root);
    }
}

/* the number of elements to pull on a GET */
#define AMOUNT 512

void shmem_threaded_bcast64 (void * target, const void *source, size_t nelems,
                             int PE_root, int PE_start, int logPE_stride, int PE_size,
                             long *pSync)
{
    const int me = shmem_my_pe();
    const int step = 1 << logPE_stride;
    const int root = (PE_root * step) + PE_start;
    int j = 0;
    int flag_uneq = 0;

    if (nelems % 20 != 0) 
      flag_uneq = 1;

    shmem_barrier (PE_start, logPE_stride, PE_size, pSync);
    if (me != root) {
        #pragma omp parallel for schedule(static) num_threads(20) 
        for (j = 0; j < nelems; j += AMOUNT) {
            int tid = omp_get_thread_num();
            if ((nelems - j) >= AMOUNT) {
                shmem_ctx_long_get(ctx[tid], target + j, source + j, AMOUNT, root);
            } else {
                shmem_ctx_long_get(ctx[tid], target + j, source + j, nelems - j, root);
            }
        }
    }
}

int main(void) {
    long * x;
    long * y;
    int mype;
    int n_pes;
    int i;
    long * pSync1;
    long * pSync2;
    double end, start;
    int provided;
    int tid;
    cpu_mhz = get_cpu_mhz(0);

    shmem_init_thread(SHMEM_THREAD_MULTIPLE, &provided);
    mype = shmem_my_pe();
    n_pes = shmem_n_pes();

    x = (long *) shmem_malloc(sizeof(long) * SIZE);
    y = (long *) shmem_malloc(sizeof(long) * SIZE);

    memset(y, 0, sizeof(long) * SIZE);

    #pragma omp parallel num_threads(40)
    {   
        tid = omp_get_thread_num();
        shmem_ctx_create(&ctx[tid]);
    }

    for (i = 0; i < SIZE; i++) {
        x[i] = i;
    }

    pSync1 = (long *) shmem_malloc(sizeof(long) * SHMEM_BCAST_SYNC_SIZE);
    pSync2 = (long *) shmem_malloc(sizeof(long) * SHMEM_BCAST_SYNC_SIZE);
    for (i = 0; i < SHMEM_BARRIER_SYNC_SIZE; i++) {
        pSync1[i] = SHMEM_SYNC_VALUE;
        pSync2[i] = SHMEM_SYNC_VALUE;
    }
    if(mype == 1){
      printf("Number of PEs: %d\n", n_pes);
      printf("Size \t\t Single \t\t Multithreaded \t\t Speedup\n");
    }

    for (i = 1; i < SIZE; i *= 2) {
        int j = 0;
        double total_time_s = 9999999.0, total_time_t = 9999999.0; 

        /* take the minimum of 100 iterations */
        for (j = 0; j < 100; j++) {
            memset(y, 0, sizeof(long) * SIZE);
            shmem_barrier_all();
            start = shmem_wtime();
            my_bcast64(y, x, i, 0, 0, 0, n_pes, pSync1);
            end = shmem_wtime();
            if ((end - start) / cpu_mhz < total_time_s) {
                total_time_s = (end - start) / cpu_mhz;
            }
        }

        for (j = 0; j < 100; j++) {
            memset(y, 0, sizeof(long) * SIZE);
            shmem_barrier_all();
            start = shmem_wtime();
            shmem_threaded_bcast64(y, x, i, 0, 0, 0, n_pes, pSync2);
            end = shmem_wtime();
            if ((end - start) / cpu_mhz < total_time_t) {
                total_time_t = (end - start) / cpu_mhz;
            }
        }
        if (mype == 1) {
            printf("%d \t\t %1.10g \t\t %0.10g \t\t %0.10g\n", i*sizeof(long), total_time_s , total_time_t, (total_time_s / total_time_t));
        }
    }
        
    shmem_finalize();
    return 0;
}

