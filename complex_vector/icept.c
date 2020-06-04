
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
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "v.h"

/* compute the solutions to the complex coefficient quadratic
 * described in our notes on page 6.
 *
 * We shall get the solutions to a complex quadratic polynomial.
 * see https://en.wikipedia.org/wiki/Complex_quadratic_polynomial
 */
int icept( cplex_type res[2],
           int *intercept_count,
           vec_type *sign,
           vec_type *loc,
           vec_type *axi,
           vec_type *obs_p,
           vec_type *obs_v )
{

    int soln_count;
    vec_type tmp[9];
    cplex_type c_tmp[18];

    memset( &tmp, 0x00, (size_t)(9)*sizeof(vec_type));
    memset( &c_tmp, 0x00, (size_t)(18)*sizeof(cplex_type));

    cplex_type quad_res[2];

    /* we shall form the complex cooefficients of a quadratic */
    cplex_type A, B, C;

    /* create a tmp vector of the semi_major_axi data to be used
     * in dot product calcs wherein we will have
     *
     *     < b^2 * c^2, a^2 * c^2, a^2 * b^2 >
     *
     * */
    tmp[0].x.r = axi->y.r * axi->y.r * axi->z.r * axi->z.r;
    tmp[0].y.r = axi->x.r * axi->x.r * axi->z.r * axi->z.r;
    tmp[0].z.r = axi->x.r * axi->x.r * axi->y.r * axi->y.r;

    /* product of the sign data with the intermediate axi vector
     * in tmp[0] to form a factor of our complex cooefficient A */

    tmp[1].x.r = sign->x.r * tmp[0].x.r;
    tmp[1].y.r = sign->y.r * tmp[0].y.r;
    tmp[1].z.r = sign->z.r * tmp[0].z.r;

    /* another factor of A is the complex components of the ray
     * direction squared. */

    /* TODO we need to check the MATH_OP_foo value for all of this */
    cplex_mult( &((tmp+2)->x), &(obs_v->x), &(obs_v->x) );
    cplex_mult( &((tmp+2)->y), &(obs_v->y), &(obs_v->y) );
    cplex_mult( &((tmp+2)->z), &(obs_v->z), &(obs_v->z) );

    /* now we form the first complex cooefficient from the dot
     * product of tmp[0] and tmp[2] to form A */
    cplex_vec_dot( &A, tmp, tmp+2 );

    /* moving onwards to complex cooefficient B wherein we need
     * multiple intermediate calculations */
    cplex_sub( c_tmp, &(obs_p->x), &(loc->x) );
    cplex_sub( c_tmp+1, &(obs_p->y), &(loc->y) );
    cplex_sub( c_tmp+2, &(obs_p->z), &(loc->z) );

    cplex_vec_set( tmp+3, c_tmp[0].r, c_tmp[0].i,
                          c_tmp[1].r, c_tmp[1].i,
                          c_tmp[2].r, c_tmp[2].i);

    /* now multiply each component of tmp[3] by the obs_v
     * components */
    cplex_mult( c_tmp+3, &(obs_v->x), &(tmp[3]).x );
    cplex_mult( c_tmp+4, &(obs_v->y), &(tmp[3]).y );
    cplex_mult( c_tmp+5, &(obs_v->z), &(tmp[3]).z );

    cplex_vec_set( tmp+4, c_tmp[3].r, c_tmp[3].i,
                          c_tmp[4].r, c_tmp[4].i,
                          c_tmp[5].r, c_tmp[5].i);

    /* dot product of the intermediate sign vec now */

    cplex_vec_dot( c_tmp+6, tmp, tmp+4 );
    c_tmp[7].r = 2.0; c_tmp[7].i = 0.0;
    cplex_mult( &B, c_tmp+6, c_tmp+7 );

    /* we first need to create (L-P) component squares */
    cplex_mult( c_tmp+8,  c_tmp,   c_tmp );
    cplex_mult( c_tmp+9,  c_tmp+1, c_tmp+1 );
    cplex_mult( c_tmp+10, c_tmp+2, c_tmp+2 );

    cplex_vec_set( tmp+5, c_tmp[8].r,  c_tmp[8].i,
                          c_tmp[9].r,  c_tmp[9].i,
                          c_tmp[10].r, c_tmp[10].i);

    /* now the dot product again with the intermediate signs */
    cplex_vec_dot( c_tmp+11, tmp, tmp+5 );

    /* now create a^2 * b^2 * c^2 */
    cplex_mult( c_tmp+12, &(axi->x), &(axi->x));
    cplex_mult( c_tmp+13, &(axi->y), &(axi->y));
    cplex_mult( c_tmp+14, &(axi->z), &(axi->z));
    cplex_mult( c_tmp+15, c_tmp+12, c_tmp+13);
    cplex_mult( c_tmp+16, c_tmp+15, c_tmp+14);

    cplex_sub( &C, c_tmp+11, c_tmp+16);

    /* TODO why even bother with res[] as separate from quad_res[] */
    if ( cplex_quadratic( quad_res, &soln_count,
                          &A, &B, &C ) == MATH_OP_FAIL ) {

        return MATH_OP_FAIL;

    }

    res[0].r = quad_res[0].r;
    res[0].i = quad_res[0].i;

    res[1].r = quad_res[1].r;
    res[1].i = quad_res[1].i;

    *intercept_count = soln_count;

    /***************************************************************
    printf ("\n---------- in icept.c we see roots ----------\n");
    printf ("     : intercept_count = %i\n", soln_count);
    printf ("   0 : res[0] = ( %-+18.12e, %-+18.12e )\n",
                                               res[0].r, res[0].i);

    printf ("   1 : res[1] = ( %-+18.12e, %-+18.12e )\n",
                                               res[1].r, res[1].i);
    printf ("-----------------------------------------------\n");
    ****************************************************************/

    return MATH_OP_SUCCESS;

}

