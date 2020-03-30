
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

#include <errno.h>
#include <locale.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "q.h"

/* this is an external custom written function that will
 * output the basic system information such as machine name
 * and operating system and kernel and memory etc. */
int sysinfo(void);

/* we need some sort of a worker thread that does at least
 * something that looks like work. It should take in a
 * pointer to a qork queue. */
void do_some_array_thing ( q_type *work_q );

int main(int argc, char **argv) {

    int candidate_int, num_pthreads;

    /* locally in this code block we can keep around some
     * integer length of the queue but to be honest we
     * should ask the queue what its length is. This means
     * a method or function should exist that will safely
     * get that data for us from the queue structure. */
    int q_len;

    struct timespec now_time;

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
        if ( ( candidate_int < 1 ) || ( candidate_int > 256 ) ){
            fprintf(stderr,"WARN : num_pthreads is unreasonable\n");
            fprintf(stderr,"     : we shall assume 4 pthreads and proceed.\n");
            num_pthreads = 4;
        } else {
            num_pthreads = candidate_int;
            fprintf(stderr,"INFO : num_pthreads is %i\n", num_pthreads);
        }
    }

    /* create our custom queue for holding task information */
    printf ( "INFO : about to call q_create()\n");
    q_type *my_q = q_create();
    printf ( "DBUG : my_q now exists at %p\n\n", my_q);

    /* make work out of think air */
    thread_parm_t *make_work = calloc( (size_t) 1, (size_t)sizeof(thread_parm_t) );

    printf ( "INFO : make_work0 is at %p\n", str0 );

    q_push( my_q, (void *)make_work );
    printf ( "INFO : q_push(make_work) done\n" );
    printf ( "     : my_q->length = %i\n\n", my_q->length );

    printf ( "     : about to call q_destroy(my_q)\n");
    printf ( "     : q_destroy(my_q) says %i items were thrown away\n",
                                                     q_destroy(my_q) );

    return ( EXIT_SUCCESS );

}

void do_some_array_thing ( q_type *work_q ) {

    int j, k;
    /* given that the queue is a blocking type of list
     * where no thread can work until something exists
     * in the list .. we can just try to pop something
     * out of it */
    thread_parm_t *foo = (thread_parm_t *)q_pop( work_q );

    /*  stuff in the parameter struct 
     *  uint32_t  t_num;
     *  double    ret_val;
     * uint64_t *big_array;
     */

    /* lets calloc a bucket of memory for the big_array */
    foo->big_array = calloc( (size_t)1048576, (size_t)sizeof(uint64_t) );
    /* maybe check if that calloc actually worked */

    for ( j=0; j<1048576; j++ ) {
        *((foo->big_array)+j) = j + 123456789;
    }

    for ( k=0; k<1024; k++ ) {
        *((foo->big_array)+(k * 256)) = k;
    }

    /* gee .. throw that away */
    free( foo->big_array );

}

