
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

#include <stdio.h>
#include <stdlib.h>

#include "v.h"

int check_dot(cplex_type *dat)
{
    /* check for a bizarre complex result from dot product */
    if ( !(dat->i == 0.0) ) {
        fprintf(stderr,"FAIL : bizarre complex dot product");
        fprintf(stderr,"     :  = ( %-+20.14e, %-+20.14e )\n",
                              dat->r, dat->i );
        return ( EXIT_FAILURE );
    }
    return ( EXIT_SUCCESS );
}

