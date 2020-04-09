
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

uint64_t fib(uint64_t n);

void *do_some_array_thing ( void *work_q ) {
    int j, k, work_counter = 0;

    pthread_t this_thread_id;

    char tbuf[32] = "";
    char fbuf[32] = "";

    /* given that the queue is a blocking type of list
     * where no thread can work until something exists
     * in the list .. we can just try to get something
     * out of the queue. Note that this will block and
     * wait for actual work to be in the queue due to
     * a pthread condition variable that we put into the
     * queue. */
    thread_parm_t *foo = (thread_parm_t *)dequeue( (q_type *)work_q );

    while ( foo ) {
        /* we have some work to do .. yay for us */
        work_counter = work_counter + 1;

        /* we need a thread safe way to say hello */
        k = sprintf( tbuf, "\nthread %3i has work item %2i\n",
                                    foo->work_num, work_counter );

        puts( tbuf );

        /* lets calloc foo->array_cnt uint64_t elements in big_array */
        foo->big_array = calloc( foo->array_cnt, (size_t)sizeof(uint64_t));
        if ( foo->big_array == NULL ) {
            /* really? possible ENOMEM? */
            if ( errno == ENOMEM ) {
                /* TODO : this is not a thread safe way to output */
                fprintf(stderr,"FAIL : calloc returns ENOMEM at %s:%d\n",
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
            *((foo->big_array)+j) = (uint64_t)j + ( (uint64_t)123456789 )
                                                   + (uint64_t)foo->fibber;
        }

        k = sprintf( fbuf,
               "\nthread %3i compute fib(%-3" PRIu64 ") = %12" PRIu64 "\n",
                            foo->work_num, foo->fibber, fib(foo->fibber) );

        puts( fbuf );

        /* gee .. throw that away */
        free( foo->big_array );
        foo->big_array = NULL;

        free( foo );
        foo = NULL;

        /* before even looking if there is new work and being blocked
         * on the condition it may be reasonable to check if we were
         * signaled to close down and exit cleanly.
         *
         *
         * who am I ? 
         *
         * int pthread_equal(pthread_t t1, pthread_t t2);
         *
         * DESCRIPTION
         *      The pthread_equal() function compares the
         *      thread IDs t1 and t2.
         *
         * RETURN VALUES
         *      The pthread_equal() function will return non-zero
         *      if the thread IDs t1 and t2 correspond to the same
         *      thread, otherwise it will return zero.
         *
         *
         * pthread_t pthread_self(void);
         *
         * DESCRIPTION
         *      The pthread_self() function returns the thread ID
         *      of the calling thread.
         *
         * RETURN VALUES
         *      The pthread_self() function returns the thread ID
         *      of the calling thread.
         */

        /* walk the entire thread collection to find this threads
         * actual id      pthread_t this_thread_id;    */
        for ( k = 0; k < THREAD_LIMIT; k++ ) {
            this_thread_id = pthread_self();
            if ( pthread_equal( worker_thread[k], this_thread_id ) ) {
                /* okay we found out out thread id number */
                sprintf( fbuf, "\nDBUG : my thread id is %3i\n", k );
                puts( fbuf );
                /* now check the flag to see if work should 
                 * continue for this thread */
            }
        }

        foo = (thread_parm_t *)dequeue( (q_type *)work_q );

    }

    return (NULL);

}

