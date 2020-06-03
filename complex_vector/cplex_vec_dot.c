
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

/* NOTE we need to avoid any attempt to handle hermitian angles
 *      in complex C3 space and thus an imaginary component
 *      will result in a MATH_OP_FAIL */

int cplex_vec_dot( cplex_type *res, vec_type *op1, vec_type *op2 )
{

    if ( (cplex_vec_check(op1) == MATH_OP_FAIL)
         ||
         (cplex_vec_check(op2) == MATH_OP_FAIL) ) {

        return MATH_OP_FAIL;

    }

    cplex_type tmp[3];

    if ( cplex_mult( &tmp[0], &op1->x, &op2->x ) == MATH_OP_FAIL) {
        return MATH_OP_FAIL;
    }

    if ( cplex_mult( &tmp[1], &op1->y, &op2->y ) == MATH_OP_FAIL) {
        return MATH_OP_FAIL;
    }

    if ( cplex_mult( &tmp[2], &op1->z, &op2->z ) == MATH_OP_FAIL) {
        return MATH_OP_FAIL;
    }


    res->r = tmp[0].r + tmp[1].r + tmp[2].r;
    res->i = tmp[0].i + tmp[1].i + tmp[2].i;

    /* For most computation we want real results only
     * however in general a vector dot product is 
     * complex in nature if the vector inputs
     * are also complex. 
     *
     * Thus we need to test if the input vectors were
     * in the complex space.
     */

    /* NOTE : we can not use the RT_EPSILON check here */
    if ( ( op1->x.i == 0 ) && ( op2->x.i == 0 )
            &&
         ( op1->y.i == 0 ) && ( op2->y.i == 0 )
            &&
         ( op1->z.i == 0 ) && ( op2->z.i == 0 ) ) {

        /* we have pure real space vector inputs
         * and this we check the result for a pure real space */
        if ( res->i == 0.0 ) {
            return MATH_OP_SUCCESS;
        } else {
            /* there must not be any imaginary component */
            return MATH_OP_FAIL;
        }

    }

    return MATH_OP_SUCCESS;

}

