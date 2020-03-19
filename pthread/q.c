
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
int sysinfo();

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

    char *str0 = calloc( (size_t) 32, (size_t)sizeof(unsigned char) );
    printf ( "INFO : string str0 is at %p\n", str0 );
    strncpy ( str0, "This is some string in str0", (size_t)27 );

    q_len = q_push( my_q, (void *)str0 );
    printf ( "INFO : q_push(str0) returned %i\n", q_len );
    printf ( "     : my_q->length = %i\n\n", my_q->length );

    char *str1 = calloc( (size_t) 64, (size_t)sizeof(unsigned char) );
    printf ( "INFO : string str1 is at %p\n", str1 );
    strncpy ( str1, "Feed the dead beef bad caffee in str1", (size_t)37 );

    q_len = q_push( my_q, (void *)str1 );
    printf ( "INFO : q_push() returned %i\n", q_len );
    printf ( "     : my_q->length = %i\n", my_q->length );

    /* okay lets cast a void pointer and listen to the screams */
    printf ( "     : before a pop we have my_q->length = %i\n", my_q->length );
    char *str2 = (char *)q_pop( my_q );
    printf ( "     : popped out string str2 and it is at %p\n", str2 );
    printf ( "     : str2 = \"%s\"\n", str2 );
    printf ( "     : after the pop we see my_q->length = %i\n\n", my_q->length );

    char *str3 = (char *)q_pop( my_q );
    printf ( "     : popped out string str3 and it is at %p\n", str3 );
    printf ( "     : str3 = \"%s\"\n", str3 );
    printf ( "     : my_q->length = %i\n", my_q->length );

    /* we will need a q_destroy() at some point but
     * for now we can just check if the thing is empty
     * or not. If not .. then empty it with a hammer */
    if ( my_q->head != NULL ) {
        /* traverse the list and wreck it */
        q_item *tmp = my_q->head;
        while ( tmp != NULL ) {
            /* if the payload exists then we know it is on the heap */
            if ( tmp->payload != NULL ) {
                free ( tmp->payload );
                tmp->payload = NULL;
            }

            tmp = tmp->next;
            free( my_q->head );
            my_q->head = tmp;
        }
    }
    free(my_q);
    free(str2);
    free(str3);

    return ( EXIT_SUCCESS );

}

