
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

void *dequeue( q_type *q ) {

    void *return_payload = NULL;

    /* We only care about the first item in the queue and
     * we want the payload from that first item. Looking
     * at this diagram we see queue->head->payload is what
     * we want.
     *
     *   +--------- queue -----------+
     *   |                           |
     *   |   head -->  some_item_N   |
     *   |                           |
     *   |   tail -->  some_item_X   |
     *   |                           |
     *   |   length =   3            |
     *   |                           |
     *   +---------------------------+
     *
     *   However the some_item_N looks like :
     *
     *   +----- some_item_N ---------+
     *   |                           |
     *   |    payload = a_pointer_x  |
     *   |                           |
     *   |    next ---> some_item_P  |
     *   |                           |
     *   +---------------------------+
     *
     * Once we get queue->head->payload then the item that
     * was called "some_item_N" no longer needs to exist.
     * The queue head must now point to whatever some_item_N
     * was pointing to as "next". That could even be NULL.
     */

    /* protect the queue from all other threads accessing it */
    pthread_mutex_lock ( q->mutex );

    /**********************************************************
     * check if the queue is empty and wait until it is alive *
     **********************************************************/
    while (    ( (q->length) == 0 )
            && ( (q->head) == NULL )
            && ( (q->tail) == NULL ) ) {

        /* WARNING : this is blocking.
         *
         * queue is empty so we await for it to get a task */
        pthread_cond_wait( &( q->alive ), q->mutex );

    }

    /* we now know for certain that the queue has something
     * at the head.  So get the payload that is pointed to. */
    return_payload = q->head->payload;

    /* redirect the head of the queue to point to whatever
     * was the next item, HOWEVER we need to save the
     * current pointer data to free() the memory later */
    q_item *tmp=q->head;
    q->head = tmp->next;
    q->length -= 1;

    /* did we just empty the queue of the only item? */
    if ( ( q->length == 0 ) && ( q->head == NULL ) ) {
        q->tail = NULL;
    }

    /* unlock the mutex */
    pthread_mutex_unlock ( q->mutex );

    /* free up the memory that was being used by the item
     * we just took the payload from.
     *
     * UPDATE from awesome strager who advises to keep the
     * free outside of the critical code section.
     */
    free(tmp);
    tmp = NULL;

    return ( return_payload );

}

