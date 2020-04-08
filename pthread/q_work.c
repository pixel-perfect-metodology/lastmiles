
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
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <unistd.h>

#include "q.h"
#include "do_work.h"

#define ELEMENT_COUNT_LIMIT 1073741824
#define THREAD_LIMIT 256

/* this is an external custom written function that will
 * output the basic system information such as machine name
 * and operating system and kernel and memory etc. */
int sysinfo(void);

/* we need some sort of a worker thread that does at least
 * something that looks like work. It should take in a
 * pointer to a work queue. However pthread_create expects
 * void pointers */
void *do_some_array_thing ( void *work_q );

int main(int argc, char **argv) {

    int j, candidate_int, num_pthreads;
    int pthread_err;

    /* how many elements to calloc into the arrays? */
    size_t req_element_num;

    struct timespec now_time;

    pthread_attr_t *attr = calloc( (size_t) 1, (size_t)sizeof(pthread_attr_t) );
    if ( attr == NULL ) {
        /* really? possible ENOMEM? */
        if ( errno == ENOMEM ) {
            fprintf(stderr,"FAIL : calloc returns ENOMEM at %s:%d\n",
                    __FILE__, __LINE__ );
        } else {
            fprintf(stderr,"FAIL : calloc fails at %s:%d\n",
                    __FILE__, __LINE__ );
        }
        perror("FAIL ");
        return ( EXIT_FAILURE );
    }

    pthread_t thread[THREAD_LIMIT];
    uint8_t thread_flag[THREAD_LIMIT];
    memset( &thread_flag, 0x00, (size_t)(THREAD_LIMIT)* sizeof(uint8_t));

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

    if ( argc != 3 ) {
        fprintf(stderr,"FAIL : insufficient arguments provided\n");
        fprintf(stderr,"     : usage %s num_pthreads, ",argv[0]);
        fprintf(stderr," array_cnt\n");
        return ( EXIT_FAILURE );
    } else {

        errno = 0;
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

        errno = 0;
        candidate_int = (int)strtol(argv[2], (char **)NULL, 10);
        if ( ( errno == ERANGE ) || ( errno == EINVAL ) ){
            fprintf(stderr,"FAIL : array_cnt not understood\n");
            perror("     ");
            return ( EXIT_FAILURE );
        }
        if ( ( candidate_int < 1048576 ) || ( candidate_int > ELEMENT_COUNT_LIMIT ) ){
            fprintf(stderr,"WARN : array_cnt is unreasonable\n");
            fprintf(stderr,"     : we shall assume 1048576 elements and proceed.\n");
            req_element_num = 1048576;
        } else {
            req_element_num = (size_t)candidate_int;
            fprintf(stderr,"INFO : req_element_num is %i\n", req_element_num );
        }
    }

    /* create our custom queue for holding task information */
    printf ( "INFO : about to call q_create()\n");
    q_type *my_q = q_create();
    printf ( "DBUG : my_q now exists at %p\n\n", my_q);

    thread_parm_t *make_work;
    /* make some work where the queue has more work elements
     * than we produce consumer threads */
    for ( j=0; j < ( num_pthreads * 2 ); j++ ) {
        errno = 0;
        make_work = calloc( (size_t) 1, (size_t)sizeof(thread_parm_t) );
        if ( make_work == NULL ) {
            /* really? possible ENOMEM? */
            if ( errno == ENOMEM ) {
                fprintf(stderr,"FAIL : calloc returns ENOMEM at %s:%d\n",
                        __FILE__, __LINE__ );
            } else {
                fprintf(stderr,"FAIL : calloc fails at %s:%d\n",
                        __FILE__, __LINE__ );
            }
            perror("FAIL ");
            /* TODO we need a smooth fail where we backout the previous
             * memory calloc calls if this is j>0 */
            return ( EXIT_FAILURE );
        }

        make_work->work_num = (uint32_t)j;

        /* create a random fibonacci number to compute */
        make_work->fibber = (uint8_t)( drand48() * 10 ) + (uint8_t)30;

        /* number of the uint64_t elements in the thread big_array */
        make_work->array_cnt = req_element_num;

        enqueue( my_q, (void *)make_work );
        printf ( "INFO : q_push(make_work) done\n" );
        printf ( "     : my_q->length = %i\n", my_q->length );

    }

    /* initialize attr with default attributes */
    errno = 0;
    if ( pthread_attr_init(attr) == ENOMEM ) {
        fprintf(stderr,"FAIL : ENOMEM from pthread_attr_init\n");
        perror("FAIL : ENOMEM");
        return ( EXIT_FAILURE );
    }

    /* system-wide contention or process contention?
     *
     * PTHREAD_SCOPE_PROCESS  or 
     * PTHREAD_SCOPE_SYSTEM 
     *
     * which is not documented much of anywhere that I have
     * seen .. yet. 
     */
    errno = 0;
    if ( pthread_attr_setscope( attr,
                                PTHREAD_SCOPE_SYSTEM)

            == EINVAL) {

        fprintf(stderr,"FAIL : pthread_attr_setscope\n");
        perror("FAIL : EINVAL");
        return ( EXIT_FAILURE );

    }

    /* From pthread_attr_setdetachstate : 
     *
     *    The detachstate can be set to either PTHREAD_CREATE_DETACHED
     *    or PTHREAD_CREATE_JOINABLE.
     *
     *    A value of PTHREAD_CREATE_DETACHED causes all threads created
     *    with attr to be in the detached state, whereas using a value
     *    of PTHREAD_CREATE_JOINABLE causes all threads created with
     *    attr to be in the joinable state. The default value of the
     *    detachstate attribute is PTHREAD_CREATE_JOINABLE.
     */
    errno = 0;
    if ( pthread_attr_setdetachstate( attr,
                                      PTHREAD_CREATE_DETACHED)

            == EINVAL) {

        fprintf(stderr,"FAIL : pthread_attr_setdetachstate\n");
        perror("FAIL : EINVAL");
        return ( EXIT_FAILURE );

    }

    for ( j=0; j < num_pthreads; j++ ) {
        errno = 0;
        pthread_err = pthread_create( &thread[j], attr,
                                      do_some_array_thing,
                                              (void *)my_q );
        /*
         * The pthread_create() function will fail if:
         *
         *   EAGAIN    The  system  lacked  the  necessary  resources  to
         *             create another thread, or the system-imposed limit
         *             on the  total  number  of  threads  in  a  process
         *             PTHREAD_THREADS_MAX would be exceeded.
         *
         *   EINVAL    The value specified by attr is invalid.
         *
         *   EPERM     The caller does not have appropriate permission to
         *             set the required scheduling parameters or schedul-
         *             ing policy.
         */

        if ( pthread_err == EAGAIN ) {

            fprintf(stderr,"FAIL : EAGAIN system lacked resources\n");
            perror("FAIL : EAGAIN");
            return ( EXIT_FAILURE );

        } else if ( pthread_err == EINVAL ) {

            fprintf(stderr,"FAIL : EINVAL attr is invalid\n");
            perror("FAIL : EINVAL");
            return ( EXIT_FAILURE );

        } else if ( pthread_err == EPERM ) {

           fprintf(stderr,"FAIL : EPERM permission denied\n");
           perror("FAIL : EPERM");
           return ( EXIT_FAILURE );

        }
    }

    /* did we ask if the threads were done?
     *
     * how about a nice polite sequential "join"
     */
    for ( j=0; j < num_pthreads; j++ ) {
        printf("calling for join on thread %i\n", j );
        pthread_join( thread[j], NULL );
        printf("join of thread %i is now complete\n", j );
    }

    /* Really we should check the length of the queue before we 
     * drop the hammer here and destroy whatever queue we have.
     * AT this point we have NO DAMN clue if the work in the queue
     * actually was all done. TODO check if the queue is empty. */
    printf ( "     : q_destroy(my_q) says %i items were thrown away\n",
                                                     q_destroy(my_q) );

    free( attr );
    attr = NULL;

    return ( EXIT_SUCCESS );

}

