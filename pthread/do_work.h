
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

#include <stdint.h>

/* struct to pass params to a POSIX thread */
typedef struct {
  uint32_t  work_num;  /* this is some arbitrary work order number */
  uint64_t *big_array; /* do some work and put data here */
  uint8_t   fibber;    /* horrific fibonacci number computation */
  size_t    array_cnt; /* number of elements to malloc/calloc */
} thread_parm_t;

