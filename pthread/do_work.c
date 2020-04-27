
/*********************************************************************
 * The Open Group Base Specifications Issue 6
 * IEEE Std 1003.1, 2004 Edition
 *
 *    An XSI-conforming application should ensure that the feature
 *    test macro _XOPEN_SOURCE is defined with the value 600 before
 *    inclusion of any header. This is needed to enable the
 *    functionality described in The _POSIX_C_SOURCE Feature Test
 *    Macro and in addition to enable the XSI extension.
 *
 *********************************************************************/
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include "q.h"
#include "do_work.h"

pthread_t worker_thread[THREAD_LIMIT];
int work_flag[THREAD_LIMIT];

uint64_t fib(uint64_t n);

void *do_some_array_thing ( void *work_q ) {

    int j, k;
    int thread_id = 0;
    int work_counter = 0;
    char tbuf[32] = "";
    char fbuf[64] = "";

    q_type *the_q = (q_type *)work_q;

    pthread_t this_thread_id;

    thread_parm_t *foo = NULL;

    /* What thread id is this ?
     *
     * Walk the entire thread collection to find this
     * thread id which is of datatype pthread_t this_thread_id
     */
    for ( k = 0; k < THREAD_LIMIT; k++ ) {
        this_thread_id = pthread_self();
        if ( pthread_equal( worker_thread[k], this_thread_id ) ) {
            /* okay we found our thread id number */
            thread_id = k;

            sprintf( tbuf, "\nDBUG : this thread id is %3i\n",
                                                           thread_id );

            puts( tbuf );

        }
    }

    /* given that the queue is a blocking type of list
     * where no thread can work until something exists
     * in the list .. we can just try to get something
     * out of the queue. Note that this will block and
     * wait for actual work to be in the queue due to
     * a pthread condition variable that we put into the
     * queue. Before we get into a blocking situation we
     * check the work_flag for this thread and see if if
     * is set to 0 in which case we bail out cleanly. */

    /* TODO think about a mutex lock to read this flag */
    if ( work_flag[thread_id] == 0 ) goto bail_out;

    /* check if the queue is empty */
    pthread_mutex_lock ( the_q->mutex );
    if (  ( the_q->length == 0 )
         && ( the_q->head == NULL )
         && ( the_q->tail == NULL ) ) {

        /* the queue is empty and thus we bail out */
        pthread_mutex_unlock ( the_q->mutex );
        return ( NULL );

    }
    pthread_mutex_unlock ( the_q->mutex );
    /****************************************************************
     *                                                              *
     *        d a n g e r     d a n g e r     d a n g e r           *
     *                                                              *
     *        We just released the mutex lock on the queue and      *
     *        thus some other thread could consume the job in       *
     *        the queue and again we stall on the condition var     *
     *                                                              *
     ****************************************************************/
    foo = (thread_parm_t *)dequeue( (q_type *)work_q );

    while ( foo ) {

        work_counter = work_counter + 1;

        /* we need a thread safe way to say hello */
        k = sprintf( tbuf, "\nthr %3i : work %3i q_item %3i\n",
                              thread_id, work_counter, foo->work_num );

        puts( tbuf );

        /* lets calloc foo->array_cnt uint64_t elements in big_array */
        foo->big_array = calloc( foo->array_cnt,
                                 (size_t)sizeof(uint64_t));

        if ( foo->big_array == NULL ) {
            /* really? possible ENOMEM? */
            if ( errno == ENOMEM ) {
                /* TODO : this is not a thread safe way to output */
                fprintf(stderr,"FAIL : calloc ENOMEM at %s:%d\n",
                        __FILE__, __LINE__ );
            } else {
                fprintf(stderr,"FAIL : calloc fails at %s:%d\n",
                        __FILE__, __LINE__ );
            }
            perror("FAIL ");
            /* this is horrible and here we bail out */
            exit ( EXIT_FAILURE );
        }

        for ( j=0; j<(int)foo->array_cnt; j++ ) {
            *((foo->big_array)+j) = (uint64_t)j + ((uint64_t)123456789)
                                               + (uint64_t)foo->fibber;
        }

        k = sprintf( fbuf,
               "\nthr %3i : fib(%-3" PRIu64 ") = %12" PRIu64 "\n",
                             thread_id, foo->fibber, fib(foo->fibber));

        puts( fbuf );

        /* gee .. throw that away */
        free( foo->big_array );
        foo->big_array = NULL;

        free( foo );
        foo = NULL;

        /* TODO think about a mutex lock to read this flag */
        if ( work_flag[thread_id] == 0 ) goto bail_out;

        /* check again if the queue is empty */
        pthread_mutex_lock ( the_q->mutex );
        if (  ( the_q->length == 0 )
             && ( the_q->head == NULL )
             && ( the_q->tail == NULL ) ) {

            /* the queue is empty and thus we bail out */
            pthread_mutex_unlock ( the_q->mutex );
            return ( NULL );

        }
        pthread_mutex_unlock ( the_q->mutex );
        foo = (thread_parm_t *)dequeue( (q_type *)work_q );

    }

    return (NULL);

bail_out:

    if ( foo != NULL ) {
        /* we have to assume that we are throwing work away. */
        free( foo );
        foo = NULL;
    }

    return (NULL);

}

