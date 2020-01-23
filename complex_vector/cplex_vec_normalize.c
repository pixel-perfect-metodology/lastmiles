
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
#include "v.h"

/* a vector "normal" is a vector with the same direction
 * but a unit length. */
int cplex_vec_normalize( vec_type *res, vec_type *op1 )
{

    double magnitude;
    vec_type tmp;

    if ( op1 == NULL ) return ( EXIT_FAILURE );

    cplex_vec_copy( &tmp, op1);
    magnitude = cplex_vec_mag( &tmp );
    if ( magnitude < RT_EPSILON ) {
        return ( EXIT_FAILURE );
    }

    res->x.r = tmp.x.r / magnitude; res->x.i = tmp.x.i / magnitude;
    res->y.r = tmp.y.r / magnitude; res->y.i = tmp.y.i / magnitude;
    res->z.r = tmp.z.r / magnitude; res->z.i = tmp.z.i / magnitude;

    return ( 0 );

}

