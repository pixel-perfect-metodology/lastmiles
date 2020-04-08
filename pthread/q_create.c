
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

q_type *q_create() {

    int err_trap_flag = 0;

    /* make a request for a queue thing in memory and
     * for extra special fun we want that memory to be
     * "clear".  That means all zeros in it. For this we
     * need one number element of a "struct q_type" thing. */

    struct q_type *q = calloc( (size_t) 1, (size_t) sizeof(struct q_type));
    if ( q == NULL ) {
        /* really? possible ENOMEM? */
        if ( errno == ENOMEM ) {
            fprintf(stderr,"FAIL : calloc returns ENOMEM at %s:%d\n",
                    __FILE__, __LINE__ );
        } else {
            fprintf(stderr,"FAIL : calloc fails at %s:%d\n",
                    __FILE__, __LINE__ );
        }
        perror("FAIL ");
        exit ( EXIT_FAILURE );
    }

    /* make sure "head" and "tail" exist and
     * since the queue is empty we want them both to
     * be NULL pointers that point to nowhere.
     */
     q->head = NULL;
     q->tail = NULL;

    /* we know that the length of this queue is zero */
    q->length = 0;

    /* The pthread_mutex_init() function will fail if:
     *
     *    EAGAIN    The system lacked the necessary  resources  (other
     *              than memory) to initialize another mutex.
     *
     *    EBUSY     An attempt was detected to  re-initialize a robust
     *              mutex  previously  initialized  but  not  yet des-
     *              troyed. See pthread_mutexattr_setrobust_np(3C).
     *
     *    EINVAL    An attempt was detected to re-initialize a  robust
     *              mutex  previously initialized with a different set
     *              of attributes.
     *              See pthread_mutexattr_setrobust_np(3C).
     *
     *    ENOMEM    Insufficient  memory  exists  to  initialize   the
     *              mutex.
     *
     *    EPERM     The caller does not have the privilege to  perform
     *              the operation.
     *
     * The  pthread_mutex_init()  function  initializes  the  mutex
     * referenced  by  mutex  with attributes specified by attr. If
     * attr is  NULL, the default mutex attributes  are  used;  the
     * effect is the same as passing the address of a default mutex
     * attributes object. Upon successful initialization, the state
     * of the mutex becomes initialized and unlocked.
     *
     * syntax : int pthread_mutex_init(
     *                       pthread_mutex_t *restrict mutex,
     *                       const pthread_mutexattr_t *restrict attr);
     *
     */
    errno = 0;
    q->mutex = calloc( (size_t) 1, (size_t)sizeof(pthread_mutex_t));

    if ( q->mutex == NULL ) {
        if ( errno == ENOMEM ) {
            fprintf(stderr,"FAIL : calloc returns ENOMEM at %s:%d\n",
                    __FILE__, __LINE__ );
        } else {
            fprintf(stderr,"FAIL : calloc fails at %s:%d\n",
                    __FILE__, __LINE__ );
        }
        perror("FAIL ");
        exit ( EXIT_FAILURE );
    }

    errno = 0;
    q->mutex_attr = calloc( (size_t) 1, (size_t)sizeof(pthread_mutexattr_t));

    if ( q->mutex_attr == NULL ) {
        if ( errno == ENOMEM ) {
            fprintf(stderr,"FAIL : calloc returns ENOMEM at %s:%d\n",
                    __FILE__, __LINE__ );
        } else {
            fprintf(stderr,"FAIL : calloc fails at %s:%d\n",
                    __FILE__, __LINE__ );
        }
        perror("FAIL ");
        exit ( EXIT_FAILURE );
    }

    err_trap_flag = 0;
    err_trap_flag = pthread_mutexattr_init( q->mutex_attr);
    if ( err_trap_flag == ENOMEM ) {
        fprintf(stderr,"FAIL : pthread_mutexattr_init at %s:%d\n",
                __FILE__, __LINE__ );

        perror("FAIL ");
        exit ( EXIT_FAILURE );
    }

    err_trap_flag = 0;
    err_trap_flag = pthread_mutexattr_settype( q->mutex_attr, PTHREAD_MUTEX_ERRORCHECK);
    if ( err_trap_flag == EINVAL ) {
        fprintf(stderr,"FAIL : pthread_mutexattr_settype at %s:%d\n",
                __FILE__, __LINE__ );

        perror("FAIL ");
        exit ( EXIT_FAILURE );
    }

    err_trap_flag = 0;
    err_trap_flag = pthread_mutex_init( q->mutex, q->mutex_attr );
    if ( err_trap_flag != 0 ) {
        if ( err_trap_flag == EAGAIN ) {
            fprintf(stderr,"FAIL : EAGAIN");
        } else if ( err_trap_flag == EBUSY ) {
            fprintf(stderr,"FAIL : EBUSY");
        } else if ( err_trap_flag == EINVAL ) {
            fprintf(stderr,"FAIL : EINVAL");
        } else if ( err_trap_flag == ENOMEM ) {
            fprintf(stderr,"FAIL : ENOMEM");
        } else if ( err_trap_flag == EPERM ) {
            fprintf(stderr,"FAIL : EPERM");
        }

        fprintf(stderr," on pthread_mutex_init at %s:%d\n",
                __FILE__, __LINE__ );

        exit ( EXIT_FAILURE );
    }

    /* setup the alive condition as a POSIX thread "condition"
     * type thing. */
    q->alive = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

    return q;

}

