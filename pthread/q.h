
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

#include <pthread.h>

/* struct to pass params to a POSIX thread */
typedef struct {
  uint32_t  t_num;   /* this is the thread number */
  double    ret_val; /* some sort of a return value */
  uint64_t *big_array;
} thread_parm_t;

typedef struct q_type {

    struct q_item *head;
    struct q_item *tail;

   /* how many items are in the queue */
    int length;

   /* We need a way to control access to
    * this list from many places and protect
    * us from multiple accesses happening at
    * the same time. */
    pthread_mutex_t q_mutex;

    /* Is this queue live or dead?
     *
     * Here I am thinking that someday and someway
     * we need to have consumers or workers out there
     * that check if there is stuff in the queue as
     * well as a way to signal that we are shutting
     * down the whole queue.
     */
    pthread_cond_t alive;

} q_type;

typedef struct q_item {

    /* we need a way to stuff a data payload or 
     * parameter information load in this thing */
    void *payload;

    /* is there a next item in the list ? */
    struct q_item *next;

} q_item;

q_type *q_create() {

    /* make a request for a queue thing in memory and
     * for extra special fun we want that memory to be
     * "clear".  That means all zeros in it. For this we
     * need one number element of a "struct q_type" thing. */

    struct q_type *q = calloc( (size_t) 1, (size_t) sizeof(struct q_type));

    /* make sure "head" and "tail" exist and 
     * since the queue is empty we want them both to 
     * be NULL pointers that point to nowhere. 
     *
     * Special note : we already asked above that the
     * memory be all set to zero and so this really is
     * not needed.  We do it anyways for clarity reasons.
     */
     q->head = NULL;
     q->tail = NULL;

    /* we know that the length of this queue is zero */
    q->length = 0;

    /* be sure the magic mutex thing exists and is setup
     * as an initialized POSIX thread mutual exclusion lock
     * based on the macro PTHREAD_MUTEX_INITIALIZER */
    q->q_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;

    /* setup the alive condition as a POSIX thread "condition"
     * type thing. */
    q->alive = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

    /* return the shiney new empty queue */
    return q;

}

int q_destroy(q_type *q) {

    int destroyed_item_count = 0;
    q_item *tmp;

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

    free(q);

    return destroyed_item_count;

}

void q_push ( q_type *q, void *p ) {

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
     * waiting but if there are then we signal that a new
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

void *q_pop( q_type *q ) {

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
    pthread_mutex_lock ( &( q->q_mutex ) );

    /* check if the queue is empty and wait until it is alive */
    while ( ( (q->length) == 0 )
            && ( (q->head) == NULL )
            && ( (q->tail) == NULL ) ) {
    
        /* queue is empty so we await for it to get a task */
        pthread_cond_wait( &( q->alive ), &( q->q_mutex ) );

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

    /* free up the memory that was being used by the item
     * we just took the payload from */
    free(tmp);
    tmp = NULL;

    /* unlock the mutex */
    pthread_mutex_unlock ( &( q->q_mutex ) );

    return ( return_payload );
     
}

