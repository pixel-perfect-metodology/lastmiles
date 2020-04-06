
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

        for ( k=0; k<1024; k++ ) {
            *((foo->big_array)+(k * 256)) = (uint64_t) k;
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

        foo = (thread_parm_t *)dequeue( (q_type *)work_q );

    }

    return (NULL);
}

