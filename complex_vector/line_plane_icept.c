
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
#include <unistd.h>
#include <math.h>

#include "v.h"

/* Here we expect to receive some point lp0 which is the "origin"
 * of a line. We also expect a direction vector lpr which might
 * not be normalized.
 *
 * We also receive a point on a plane pl0 and a normal vector pn
 * which might not be normalized. Optionally we may receive some
 * linearly independant basis vectors plu and plv which must be
 * orthogonal to pn.  If we do not receive plu and plv then we
 * must compute some reasonable basis vectors in the plane.
 *
 * The return value will indicate success or not. In the event of
 * a successful intercept computation then we return four vectors
 * thus : 
 *
 *            icept_pt : the actual intercept point coordinates
 *
 *            plun     : normalized plane basis vector plu
 *
 *            plvn     : normalized plane basis vector plv
 *            
 *            kst      : scalar solution values for the following 
 *
 *                        icept_pt = lp0 + k * norm[ lpr ]
 *
 *                        icept_pt = pl0 + s * plun
 *                                       + t * plvn
 */

int check_dot(cplex_type *dat);

int line_plane_icept( vec_type *icept_pt,
                      vec_type *plun,
                      vec_type *plvn,
                      vec_type *kst,
                      vec_type *lp0, vec_type *lpr,
                      vec_type *pl0, vec_type *pn,
                      vec_type *plu, vec_type *plv) 
{
    int return_value = 0;
    cplex_type ctmp[12];
    vec_type i_hat, j_hat, tmp[12];
    double lpr_pn_theta;

    /* rh_col is right hand column for Cramer call with
     * res_vec as the result if it exists */
    vec_type v[4], rh_col, res_vec;


    /* It seems reasonable to check if the input data is
     * sane. At the very least we must ask if the data even
     * exists first. */
    if ( ( lp0 == NULL ) 
       ||( lpr == NULL )
       ||( pl0 == NULL )
       ||( pn  == NULL ) ) return ( return_value );

    /* need vectors of a reasonable length to work with */
    if (  ( cplex_vec_mag( lpr ) < RT_EPSILON ) 
       || ( cplex_vec_mag( pn  ) < RT_EPSILON ) )
                           return ( return_value );

    /* check if lpr and pn are at orthogonal */
    cplex_vec_dot( ctmp, lpr, pn );
    if ( cplex_mag( ctmp ) == 0.0 ) return ( return_value );

    /* we now check for an angle less than RT_ANGLE_EPSILON */
    cplex_vec_normalize( tmp, lpr );
    cplex_vec_normalize( tmp+1, pn );
    /* So norm[ lpr ] --> tmp[0]
     *    norm[ pn  ] --> tmp[1]  */

    cplex_vec_dot( ctmp, tmp, tmp+1);
    if ( check_dot( ctmp ) == EXIT_FAILURE ) return ( return_value );

    lpr_pn_theta = acos(ctmp[0].r);
    if ( fabs(lpr_pn_theta) < RT_ANGLE_EPSILON ) {
        if ( lpr_pn_theta == 0.0 ) {
            fprintf(stderr,"FAIL : lpr and pn orthogonal.\n");
        } else {
            fprintf(stderr,"FAIL : lpr and pn near orthogonal.\n");
        }
        return ( return_value );
    }

    /* Common sense checks are done and we can proceed with 
     * handling plu and plv which may not even exist.
     *
     * If plu and plv both exist then we need to check that
     * they are linearly independant and then create the 
     * normalized versions of them. If they do not exist then
     * we have the task of creation based on the existing basis
     * vectors i_hat and j_hat. We may also have the situation
     * where only one of them exists and we must compute the
     * other.
     *
     * if dot( N, i_hat ) != 1 then u = cross( N, i_hat )
     *     else we use j_hat. 
     *
     *     note that tmp[1] is norm[pn] at the moment
     * */

    cplex_vec_set ( &i_hat, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    cplex_vec_set ( &j_hat, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);

    if ( ( plu == NULL ) && ( plv == NULL ) ) {
        /* We must compute both the plu and plv where the
         * i_hat or j_hat basis vectors are used. */

        cplex_vec_dot( ctmp+1, tmp+1, &i_hat);
        if ( check_dot( ctmp+1 ) == EXIT_FAILURE )
            return ( return_value );

        if ( fabs(fabs(ctmp[1].r) - 1.0) < RT_EPSILON ) {
            /* we need to use j_hat because norm[pn] is
             * most likely near perfect parallel to i_hat */

            cplex_vec_dot( ctmp+1, tmp+1, &j_hat);
            if ( check_dot( ctmp+1 ) == EXIT_FAILURE )
                return ( return_value );

            cplex_vec_copy( tmp+2, &j_hat);
        } else {
            /* we may continue with i_hat as the reference
             * basis vector */
            cplex_vec_copy( tmp+2, &i_hat);
        }

        /* create an orthogonal vector plu in tmp+3 */
        cplex_vec_cross( tmp+3, tmp+1, tmp+2 );
        /* now we create the useful plun */
        cplex_vec_normalize( plun, tmp+3 );

        printf("dbug : u_hat = %+-16.9e    %+-16.9e    %+-16.9e\n",
                     plun->x.r, plun->y.r, plun->z.r );

        /* create an orthogonal vector plv in tmp+4 */
        cplex_vec_cross( tmp+4, tmp+1, plun );
        cplex_vec_normalize( plvn, tmp+4 );

        printf("dbug : v_hat = %+-16.9e    %+-16.9e    %+-16.9e\n",
                     plvn->x.r, plvn->y.r, plvn->z.r );

    } else {
        /* TODO : none of this will work obviously */
        /* We know that we have at least plu or plv now */
        if ( ( plu == NULL ) || ( plv == NULL ) ) {
            /* lovely, we only have one of them */
            if ( plu == NULL ) {
                /* compute plu with the existing plv */
            } else {
                /* compute plv with the existing plu */
            }
        } else {
            /* okay great we have them both and need to
             * check them and normalize them */
        }
    }


    /* lets create the column of data for P3 - P0 in our
     * diagram. This would be  pl0 - lp0. */
    cplex_vec_set ( &rh_col, pl0->x.r - lp0->x.r,
                             pl0->x.i - lp0->x.i,
                             pl0->y.r - lp0->y.r,
                             pl0->y.i - lp0->y.i,
                             pl0->z.r - lp0->z.r,
                             pl0->z.i - lp0->z.i );

    v[0].x.r = tmp[0].x.r;        v[0].x.i = tmp[0].x.i;
    v[0].y.r = plun->x.r;         v[0].y.i = plun->x.i;
    v[0].z.r = plvn->x.r;         v[0].z.i = plvn->x.i;

    v[1].x.r = tmp[0].y.r;        v[1].x.i = tmp[0].y.i;
    v[1].y.r = plun->y.r;         v[1].y.i = plun->y.i;
    v[1].z.r = plvn->y.r;         v[1].z.i = plvn->y.i;

    v[2].x.r = tmp[0].z.r;        v[2].x.i = tmp[0].z.i;
    v[2].y.r = plun->z.r;         v[2].y.i = plun->z.i;
    v[2].z.r = plvn->z.r;         v[2].z.i = plvn->z.i;

    printf("Matrix with line plane intercept data.\n");
    printf("dbug : row 1 =    %+-16.9e    %+-16.9e    %+-16.9e\n",
            v[0].x.r, v[0].y.r, v[0].z.r );

    printf("     : row 2 =    %+-16.9e    %+-16.9e    %+-16.9e\n",
            v[1].x.r, v[1].y.r, v[1].z.r );

    printf("     : row 3 =    %+-16.9e    %+-16.9e    %+-16.9e\n",
            v[2].x.r, v[2].y.r, v[2].z.r );

    cplex_det( ctmp+2, &v[0], &v[1], &v[2] ); 
    printf("\n     :   det =    %+-16.9e, %g )\n", ctmp[2].r, ctmp[2].i);

    printf("\nSolve for line plane intercept with Cramers rule.\n\n");
    if ( cplex_cramer(&res_vec, &v[0], &v[1], &v[2], &rh_col) != 0 ) {
        printf("dbug : There is no valid solution.\n");
    } else {
        if (    ( fabs(res_vec.x.i) > RT_EPSILON )
             || ( fabs(res_vec.y.i) > RT_EPSILON )
             || ( fabs(res_vec.z.i) > RT_EPSILON ) ) {
            printf("dbug : complex solution?\n");
        } else {
            /* the result will be  k , -s, -t */
            printf("    k = %+-20.16e\n", res_vec.x.r );
            printf("    s = %+-20.16e\n", -1.0 * res_vec.y.r );
            printf("    t = %+-20.16e\n\n", -1.0 * res_vec.z.r );
        }
    }

    return ( 1 );

}

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

