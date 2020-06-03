
#define _XOPEN_SOURCE 600

#include <math.h>
#include "v.h"

int cplex_vec_check( vec_type *op )
{

    if ( op == NULL ) {
        return MATH_OP_FAIL;
    }

    if ( cplex_check(&(op->x)) == MATH_OP_FAIL ) {
        return MATH_OP_FAIL;
    }

    if ( cplex_check(&(op->y)) == MATH_OP_FAIL ) {
        return MATH_OP_FAIL;
    }

    if ( cplex_check(&(op->z)) == MATH_OP_FAIL ) {
        return MATH_OP_FAIL;
    }

    return MATH_OP_SUCCESS;

}
