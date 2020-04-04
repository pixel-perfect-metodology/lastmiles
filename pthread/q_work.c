
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
 * pointer to a work queue. However pthread_create expects
 * void pointers */
void *do_some_array_thing ( void *work_q );

/* struct to pass params to a POSIX thread */
typedef struct {
  uint32_t  work_num;  /* this is some arbitrary work order number */
  int       ret_val;   /* some sort of a return value */
  uint64_t *big_array; /* do some work and put data here */
} thread_parm_t;

int main(int argc, char **argv) {

    int j, candidate_int, num_pthreads;
    int pthread_err;

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

    pthread_t thread[256];

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

    thread_parm_t *make_work;
    /* make some work out of think air */
    for ( j=0; j<num_pthreads; j++ ) {
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

        make_work->work_num = j;

        enqueue( my_q, (void *)make_work );
        printf ( "INFO : q_push(make_work) done\n" );
        printf ( "     : my_q->length = %i\n\n", my_q->length );

    }

    /* initialize attr with default attributes */
    if ( pthread_attr_init(attr) == ENOMEM ) {
        fprintf(stderr,"FAIL : ENOMEM from pthread_attr_init\n");
        perror("FAIL : ENOMEM");
        return ( EXIT_FAILURE );
    }

    /* system-wide contention */
    pthread_attr_setscope(attr, PTHREAD_SCOPE_SYSTEM);

    for ( j=0; j<num_pthreads; j++ ) {
        errno = 0;
        pthread_err = pthread_create( &thread[j], attr, do_some_array_thing, (void *)my_q );
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

    printf ( "     : about to call q_destroy(my_q)\n");
    printf ( "     : q_destroy(my_q) says %i items were thrown away\n",
                                                     q_destroy(my_q) );

    free( attr );
    attr = NULL;

    return ( EXIT_SUCCESS );

}

void *do_some_array_thing ( void *work_q ) {

    int j, k;
    /* given that the queue is a blocking type of list
     * where no thread can work until something exists
     * in the list .. we can just try to pop something
     * out of it */
    thread_parm_t *foo = (thread_parm_t *)dequeue( (q_type *)work_q );

    /* lets calloc a bucket of memory for the big_array */
    foo->big_array = calloc( (size_t)1048576, (size_t)sizeof(uint64_t) );
    if ( foo->big_array == NULL ) {
        /* really? possible ENOMEM? */
        if ( errno == ENOMEM ) {
            fprintf(stderr,"FAIL : calloc returns ENOMEM at %s:%d\n",
                    __FILE__, __LINE__ );
        } else {
            fprintf(stderr,"FAIL : calloc fails at %s:%d\n",
                    __FILE__, __LINE__ );
        }
        perror("FAIL ");
        /* this is horrible and here we bail like ass hats */
        exit ( EXIT_FAILURE );
    }

    for ( j=0; j<1048576; j++ ) {
        *((foo->big_array)+j) = (uint64_t)( j + 123456789 );
    }

    for ( k=0; k<1024; k++ ) {
        *((foo->big_array)+(k * 256)) = (uint64_t) k;
    }

    /* gee .. throw that away */
    free( foo->big_array );
    foo->big_array = NULL;

    return ( NULL );

}

