
#define _XOPEN_SOURCE 600

#include <math.h>
#include "v.h"

int cplex_check( cplex_type *op )
{

    /*
     * check for invalid data: null, nan, inf 
     *
     * The fpclassify() macro takes an argument of x and returns
     * one of the following :
     *
     *     FP_INFINITE    x is an infinite number
     *     FP_NAN         x is not a number (NaN)
     *     FP_NORMAL      x is a normalized number
     *     FP_SUBNORMAL   x is a denormalized number
     *     FP_ZERO        x is a zero value (+0 or -0)
     */

    /* It seems reasonable to check if the input data is
     * sane. At the very least we must ask if the data even
     * exists first. */
    if ( op == NULL ) {
        return MATH_OP_FAIL;
    }

    if ( ( fpclassify(op->r) == FP_NAN )
         ||
         ( fpclassify(op->i) == FP_NAN ) ) {

        return MATH_OP_FAIL;

    }

    if ( ( fpclassify(op->r) == FP_INFINITE )
         ||
         ( fpclassify(op->i) == FP_INFINITE ) ) {

        return MATH_OP_FAIL;

    }

    if ( ( fpclassify(op->r) == FP_SUBNORMAL )
         ||
         ( fpclassify(op->i) == FP_SUBNORMAL ) ) {

        return MATH_OP_FAIL;

    }

    return MATH_OP_SUCCESS;

}
