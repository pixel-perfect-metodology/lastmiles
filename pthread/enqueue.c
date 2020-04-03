
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

void enqueue ( q_type *q, void *p ) {

    /* set the mutex as locked */
    pthread_mutex_lock ( &( q->q_mutex ) );

    /* we need to create a new queue item and put
     * the payload into it */
    struct q_item *new_item = calloc((size_t) 1, (size_t)sizeof(struct q_item));
    new_item->payload = p;

    /* we used calloc to give us clear memory but to be
     * clear this item points to nowhere at the moment */
    new_item->next = NULL;

    /* Is the queue list empty? Check if head and tail
     * point nowhere OR even check if length is zero.
     *
     * To be clear the queue itself is NOT a linked
     * list but rather the items inside it are linked.
     *
     * If the queue is empty then the head points to
     * nowhere as well as the tail. The length will
     * also be zero.  If there is only a single item
     * in the queue then the head and tail both point
     * to that single item. */

    if ( ( (q->length) == 0 )
        && ( (q->head) == NULL )
        && ( (q->tail) == NULL ) ) {

        /* the queue is indeed empty.
         *
         * Just place the new_item on the head and
         * the tail and set length to one.
         */
        q->head = new_item;
        q->tail = new_item;
        q->length = 1;

    } else {

        /* The queue is not empty.
         *
         * Take this new_item and stick it on the queue
         * tail.  However we already have something on
         * the tail and we need to preserve that pointer.
         * Therefore whatever is on the tail now must
         * point to the new_item. Also the queue tail
         * will point to this new_item as it really is
         * now on the end of the list.
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
         *   +----- some_item_P ---------+
         *   |                           |
         *   |    payload = a_pointer_y  |
         *   |                           |
         *   |    next ---> some_item_X  |
         *   |                           |
         *   +---------------------------+
         *
         *   +----- some_item_X ---------+
         *   |                           |
         *   |    payload = a_pointer_z  |
         *   |                           |
         *   |    next ---> NULL         |
         *   |      This "next" is also  |
         *   |      queue->tail->next    |
         *   |                           |
         *   +---------------------------+
         *
         */
        q->tail->next = new_item;
        q->tail = new_item;
        q->length += 1;

    }

    /* unlock the mutex */
    pthread_mutex_unlock ( &( q->q_mutex ) );

    /* send out a signal to at least one thread consumer
     * which may be waiting. No promise anything is actually
     * waiting but if there is then we signal that a new
     * task has arrived.
     *
     * From the manpage :
     *
     *    The pthread_cond_signal() call unblocks at least one
     *    of the threads that are blocked on the specified
     *    condition variable condition. This is if any threads
     *    are blocked on cond.
     *
     */
    pthread_cond_signal( &( q->alive ) );

}

