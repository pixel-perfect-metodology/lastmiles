
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
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "q.h"

int q_destroy(q_type *q) {

    int err_trap_flag, destroyed_item_count = 0;
    q_item *tmp;

    /* set the mutex as locked */
    pthread_mutex_lock ( q->mutex );

    if ( q->head != NULL ) {
        /* traverse the list and free items as we hit them */
        tmp = q->head;
        while ( tmp != NULL ) {
            /* if the payload exists then free it */
            if ( tmp->payload != NULL ) {
                free ( tmp->payload );
                tmp->payload = NULL;
                destroyed_item_count += 1;
            }

            tmp = tmp->next;
            free( q->head );
            q->head = tmp;

        }
    }

    /* TODO destroy the condition variable in the queue,
     *
     * The pthread_cond_destroy() function may fail if:
     *
     *     EBUSY   The implementation has detected  an  attempt
     *             to  destroy  the  object  referenced by cond
     *             while it is referenced (for  example,  while
     *             being   used  in  a  pthread_cond_wait()  or
     *             pthread_cond_timedwait()) by another thread.
     *
     *     EINVAL  The value specified by cond is invalid.
     *
     */

    /* unlock the mutex */
    pthread_mutex_unlock ( q->mutex );

    /* destroy the mutex and the mutex attribute */
    err_trap_flag = 0;
    err_trap_flag = pthread_mutex_destroy( q->mutex );
    if ( err_trap_flag == EBUSY ) {
        fprintf(stderr,"FAIL : EBUSY from pthread_mutex_destroy at %s:%d\n",
                        __FILE__, __LINE__ );
    } else if ( err_trap_flag == EINVAL ) {
        fprintf(stderr,"FAIL : EINVAL in pthread_mutex_destroy at %s:%d\n",
                        __FILE__, __LINE__ );
    }

    err_trap_flag = 0;
    err_trap_flag = pthread_mutexattr_destroy( q->mutex_attr);
    if ( err_trap_flag == EINVAL ) {
        fprintf(stderr,"FAIL : EINVAL in pthread_mutexattr_destroy at %s:%d\n",
                        __FILE__, __LINE__ );
    }

    free(q->mutex);
    q->mutex = NULL;
    free(q->mutex_attr);
    q->mutex_attr = NULL;

    /* we may be wrecking havok here with threads that
     * are awaiting the queue to be available */
    free(q);

    return destroyed_item_count;

}

