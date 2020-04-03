
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

q_type *q_create(void);
int q_destroy(q_type *q);
void enqueue ( q_type *q, void *p );
void *dequeue( q_type *q );

