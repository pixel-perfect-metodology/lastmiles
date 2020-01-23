
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

#include <math.h>
#include "v.h"

/* initialize the values inside a complex vector to zero */
int cplex_vec_null( vec_type *op )
{
    op->x.r = 0.0; op->x.i = 0.0;
    op->y.r = 0.0; op->y.i = 0.0;
    op->z.r = 0.0; op->z.i = 0.0;
    return ( EXIT_SUCCESS );
}

