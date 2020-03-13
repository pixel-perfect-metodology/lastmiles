
#define _XOPEN_SOURCE 600

#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <unistd.h>

#include "p0.h"

int sysinfo(void);
int ru(void);

void *big_array_fill(void *); /* thread routine */

/* array of thread IDs */
pthread_t tid[256];

int main(int argc, char **argv)
{

    int    i, j;
    int    candidate_int, num_pthreads;
    struct timespec now_time;

    /* some reasonable upper limit */
    thread_parm_t *parm[256];

    setlocale( LC_ALL, "C" );
    sysinfo();

    /* Get the REALTIME_CLOCK time in a timespec struct */
    if ( clock_gettime( CLOCK_REALTIME, &now_time ) == -1 ) {
        /* We could not get the clock. Bail out. */
        fprintf(stderr,"ERROR : could not attain CLOCK_REALTIME\n");
        return(EXIT_FAILURE);
    } else {
        /* call srand48() with the sub-second time data */
        srand48( (long) now_time.tv_nsec );
    }

    errno = 0;
    if ( argc != 2 ) {
        fprintf(stderr,"FAIL : insufficient arguments provided\n");
        fprintf(stderr,"     : usage %s num_pthreads\n",argv[0]);
        return ( EXIT_FAILURE );
    } else {
        candidate_int = (int)strtol(argv[1], (char **)NULL, 10);
        if ( ( errno == ERANGE ) || ( errno == EINVAL ) ){
            fprintf(stderr,"FAIL : num_pthreads not understood\n");
            perror("     ");
            return ( EXIT_FAILURE );
        }
        if ( ( candidate_int < 2 ) || ( candidate_int > 256 ) ){
            fprintf(stderr,"WARN : num_pthreads is unreasonable\n");
            fprintf(stderr,"     : we shall assume 4 pthreads and proceed.\n");
            num_pthreads = 4;
        } else {
            num_pthreads = candidate_int;
            fprintf(stderr,"INFO : num_pthreads is %i\n", num_pthreads);
        }
    }

    printf("\n-------------- begin dispatch -----------------------\n");

    errno = 0;
    for ( i = 0; i < num_pthreads; i++) {
        parm[i] = calloc( (size_t) 1 , (size_t) sizeof(thread_parm_t) );

        if ( parm[i] == NULL ) {
            if ( errno == ENOMEM ) {
                fprintf(stderr,"FAIL : calloc returns ENOMEM at %s:%d\n",
                        __FILE__, __LINE__ );
            } else {
                fprintf(stderr,"FAIL : calloc fails at %s:%d\n",
                        __FILE__, __LINE__ );
            }
            perror("FAIL ");
            /* gee .. before we bail out did we allocate any of the
             * previous thread parameter memory regions? If so then
             * clean up before bailing out. In fact we may have 
             * already dispatched out threads. */

            if (i == 0 ) {
                /* the system gave us no resources at all */
                fprintf(stderr,"FAIL : we are unable to even begin\n");
                return ( EXIT_FAILURE );
            }

            for ( j = 0; j < i; j++ ) {
                /* lets ask those threads to just be nice and 
                 * we call them in with a join */
                pthread_join(tid[j], NULL);

                fprintf(stderr,"BAIL : pthread_join(%i) done.\n", j);

                fprintf(stderr,"     : thread %i returned %-14.12g data.\n",
                                                      j, parm[j]->ret_val );

                free(parm[j]);
                /* belt and suspenders safety we set the pointer to NULL */
                parm[j] = NULL;
            }
            fprintf(stderr,"BAIL : cleanup done.\n", j);

            /* We have an opportunity here where we could dispatch
             * out the remaining threads in chunks of no more
             * than ( i - 1 ) at a time. If we get a calloc problem
             * then we just bail out. */

            ru();

            return ( EXIT_FAILURE );

        }

        parm[i]->t_num = (uint32_t)i;
        parm[i]->sleep_time = 1 + (int)( drand48() * 10.0 );

        pthread_create( &tid[i], NULL, big_array_fill, (void *)parm[i] );

        printf("INFO : pthread_create %2i called.\n", i );
    }
    printf("\n-------------- end dispatch -------------------------\n");

    for ( i = 0; i < num_pthreads; i++) {
        pthread_join(tid[i], NULL);

        printf("main() pthread_join %2i returned %-14.12g data.\n",
                                                  i, parm[i]->ret_val );

    }
    printf("\n-------------- end join -----------------------------\n");
    printf("All %i threads have terminated\n", i);

    for ( i = 0; i < num_pthreads; i++) {
        free(parm[i]);
        parm[i] = NULL;
    }

    ru();

    return ( EXIT_SUCCESS );

}

void *big_array_fill(void *recv_parm)
{
    thread_parm_t *p = (thread_parm_t *)recv_parm;

    printf("TRD  : %d filling the big_array.\n", p->t_num);
    for ( p->loop0 = 0; p->loop0 < BIG_ARRAY_DIM0; p->loop0++ ) {
        for ( p->loop1 = 0; p->loop1 < BIG_ARRAY_DIM1; p->loop1++ ) {
            p->big_array[p->loop0][p->loop1] = (uint64_t)(p->loop0 * p->loop1 + p->t_num + 1);
        }
    }
    printf("TRD  : %d big_array full.\n", p->t_num);

    /* return some random data */
    p->ret_val = drand48();

    return (NULL);
}

