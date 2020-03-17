
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
#include "q.h"

int main(int argc, char **argv) {

    int foo;

    printf ( "hellow you threaded world you.\n");

    q_type *my_q = q_create();

    char *barf = calloc( (size_t) 32, (size_t)sizeof(unsigned char) );
    printf ( "INFO :  barf string is at %p\n", barf );

    foo = q_push( my_q, (void *)barf );
    printf ( "INFO : q_push() returned %i\n", foo );
    printf ( "     : my_q->length = %i\n", my_q->length );

    char *gag = calloc( (size_t) 64, (size_t)sizeof(unsigned char) );
    printf ( "INFO :   gag string is at %p\n", gag );

    foo = q_push( my_q, (void *)gag );
    printf ( "INFO : q_push() returned %i\n", foo );
    printf ( "     : my_q->length = %i\n", my_q->length );

    /* okay lets cast a void pointer and listen to the screams */
    char *puke = (char *)q_pop( my_q );
    printf ( "     : my_q->length = %i\n", my_q->length );
    printf ( "     :  puke string is at %p\n", puke );
    printf ( "     : my_q->length = %i\n", my_q->length );

    char *fart = (char *)q_pop( my_q );
    printf ( "     : my_q->length = %i\n", my_q->length );
    printf ( "     :  fart string is at %p\n", fart );
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
    my_q = NULL;

    return ( EXIT_SUCCESS );

}

