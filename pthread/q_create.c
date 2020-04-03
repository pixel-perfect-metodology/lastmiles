
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

#include <stdlib.h>
#include <pthread.h>
#include "q.h"

q_type *q_create() {

    /* make a request for a queue thing in memory and
     * for extra special fun we want that memory to be
     * "clear".  That means all zeros in it. For this we
     * need one number element of a "struct q_type" thing. */

    struct q_type *q = calloc( (size_t) 1, (size_t) sizeof(struct q_type));

    /* make sure "head" and "tail" exist and
     * since the queue is empty we want them both to
     * be NULL pointers that point to nowhere.
     */
     q->head = NULL;
     q->tail = NULL;

    /* we know that the length of this queue is zero */
    q->length = 0;

    /* mutex setup as an initialized POSIX thread mutual
     * exclusion lock based on the macro PTHREAD_MUTEX_INITIALIZER */
    q->q_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;

    /* setup the alive condition as a POSIX thread "condition"
     * type thing. */
    q->alive = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

    return q;

}

