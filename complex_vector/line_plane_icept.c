
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

int line_plane_icept( vec_type *icept_pt,
                      vec_type *plun,
                      vec_type *plvn,
                      vec_type *kst,
                      vec_type *lp0, vec_type *lpr,
                      vec_type *pl0, vec_type *pn,
                      vec_type *plu, vec_type *plv) 
{
    int status, return_value = 0;
    cplex_type ctmp[12];
    vec_type i_hat, j_hat, tmp[12];
    double lpr_pn_theta, u_mag, v_mag;

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

    /* check if lpr and pn are orthogonal */
    cplex_vec_dot( ctmp, lpr, pn );
    if ( check_dot( ctmp ) == EXIT_FAILURE ) return ( return_value );
    if ( ctmp->r == 0.0 ) return ( return_value );

    /* we now check for an angle less than RT_ANGLE_EPSILON */
    status = cplex_vec_normalize( tmp, lpr );
    if ( status == EXIT_FAILURE ) return ( return_value );

    status = cplex_vec_normalize( tmp+1, pn );
    if ( status == EXIT_FAILURE ) return ( return_value );
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

uv:     cplex_vec_dot( ctmp+1, tmp+1, &i_hat);
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
        status = cplex_vec_normalize( plun, tmp+3 );
        if ( status == EXIT_FAILURE ) return ( return_value );

        /* create an orthogonal vector plv in tmp+4 */
        cplex_vec_cross( tmp+4, tmp+1, plun );
        status = cplex_vec_normalize( plvn, tmp+4 );
        if ( status == EXIT_FAILURE ) return ( return_value );

    } else {

        /* We know that we have at least plu or plv now */
        if ( ( plu == NULL ) || ( plv == NULL ) ) {
            /* lovely, we only have one of them */
            if ( plu == NULL ) {
                /* compute plu with the existing plv */
            } else {
                /* compute plv with the existing plu */
            }
        } else {
            /* Both are non-null pointers.
             * Do we have a reasonable magnitude? */
            u_mag = cplex_vec_mag( plu );
            v_mag = cplex_vec_mag( plv );

            /* We may need to deal with very small planes in the event
             * of triangle point tesselation. Objects such as the UTAH
             * teapot may have many very small triangle sections and 
             * each will define a plane intercept issue. With very 
             * tiny triangles we may have very tiny vectors u and v as
             * well as plane normals. */

            if (    ( u_mag < RT_EPSILON )
                 || ( v_mag < RT_EPSILON ) ) {

                /* One or both are very small. Is either zero ? */
                if ( ( u_mag == 0.0 ) || ( v_mag == 0.0 ) ) {
                    /* Well one of them is zero magnitude.
                     * Do we have a u vector ? */

                    if ( u_mag == 0.0 ) {

                        /* u vector is zero so lets ask about v and
                         * see if it is reasonable. Otherwise just
                         * compute them both as above. */
                        if ( ( v_mag == 0.0 ) || ( v_mag < RT_EPSILON ) ) goto uv;

                        /* normalize plv */
                        status = cplex_vec_normalize( plvn, plv );
                        if ( status == EXIT_FAILURE ) return ( return_value );

                        /* check if plvn is orthogonal to pn and
                         * if not then start over. Bear in mind that
                         * we have no idea if pn is normalized and 
                         * thus we must use tmp[1] from above which is
                         * the plane normal actually normalized. */

                        cplex_vec_dot( ctmp, tmp+1, plvn );
                        if ( check_dot( ctmp ) == EXIT_FAILURE )
                            return ( return_value );

                        if ( ctmp->r != 0.0 ) goto uv;

                        /* we have a valid plvn and may compute plu with
                         * a cross product of pn and plvn */
                        cplex_vec_cross( tmp+3, pn, plvn );
                        /* normalize that into plun */
                        status = cplex_vec_normalize( plun, tmp+3 );
                        if ( status == EXIT_FAILURE ) return ( return_value );

                    } else if ( v_mag == 0.0 ) {

                        /* v vector is zero. same situation as above
                         * we need to check u vector and then decide
                         * if we need to abandon the compute here and
                         * merely re-compute both u and v vectors. */
                        if ( ( u_mag == 0.0 ) || ( u_mag < RT_EPSILON ) ) goto uv;

                        /* normalize plu  */
                        status = cplex_vec_normalize( plun, plu );
                        if ( status == EXIT_FAILURE ) return ( return_value );

                        /* check if plun is orthogonal to pn. 
                         * here we use tmp[1] which is pn normalized */
                        cplex_vec_dot( ctmp, tmp+1, plun );
                        if ( check_dot( ctmp ) == EXIT_FAILURE )
                            return ( return_value );

                        if ( ctmp->r != 0.0 ) goto uv;

                        /* we have a valid plun and may compute plv
                         * with a cross product of pn and plun */
                        cplex_vec_cross( tmp+3, pn, plun );
                        /* normalize that into plvn */
                        status = cplex_vec_normalize( plvn, tmp+3 );
                        if ( status == EXIT_FAILURE ) return ( return_value );

                    } else {

                        /* Both u and v are zero magnitude. So
                         * just start over and compute them both. */
                        goto uv;

                    }
                } else {
                    /* Neither u nor v is zero in size but at least
                     * one of them is tiny. Smaller than RT_EPSILON. */
                }
            }
        }
    }

    printf("dbug : u_hat = %+-16.9e", plun->x.r);
    printf("    %+-16.9e", plun->y.r);
    printf("    %+-16.9e\n", plun->z.r );
    printf("dbug : v_hat = %+-16.9e", plvn->x.r);
    printf("    %+-16.9e", plvn->y.r);
    printf("    %+-16.9e\n", plvn->z.r );

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
    printf("     :   det =    %+-16.9e\n", ctmp[2].r);

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

    /* copy the result data into rst */
    cplex_vec_set ( kst, res_vec.x.r, res_vec.x.i,
                         -1.0 * res_vec.y.r, res_vec.y.i,
                         -1.0 * res_vec.z.r, res_vec.z.i);


    /* We can compute the actual intercept point two ways :
     *
     *     icept_pt = lp0 + k * norm[ lpr ]
     *
     *     icept_pt = pl0 + s * plun
     *                    + t * plvn
     *
     * It seems reasonable to try both and then compare the
     * results and verify they are within RT_EPSILON of each
     * other. TODO : think about the wisdom of this.
     */

    /* multiply     k * norm[ lpr ]     */
    cplex_vec_scale( tmp+5, tmp, kst->x.r);
    cplex_vec_add( tmp+6, lp0, tmp+5 );
    printf("\n    icept_pt = lp0 + k * norm[ lpr ]\n");
    printf("             = < %+-16.9e, %+-16.9e, %+-16.9e >\n",
                          tmp[6].x.r, tmp[6].y.r, tmp[6].z.r );

    /* multiply     s * plun     */
    cplex_vec_scale( tmp+7, plun, kst->y.r);
    /* multiply     t * plvn     */
    cplex_vec_scale( tmp+8, plvn, kst->z.r);
    /* sum them up with pl0 */
    cplex_vec_add( tmp+9, pl0, tmp+7);
    cplex_vec_add( tmp+10, tmp+9, tmp+8);
    printf("\n    icept_pt = pl0 + s * plun + t * plvn\n");
    printf("             = < %+-16.9e, %+-16.9e, %+-16.9e >\n",
                       tmp[10].x.r, tmp[10].y.r, tmp[10].z.r );

    /* check if we are within RT_EPSILON */
    if ( ( fabs( tmp[6].x.r - tmp[10].x.r ) < RT_EPSILON )
            ||
         ( fabs( tmp[6].y.r - tmp[10].y.r ) < RT_EPSILON )
            ||
         ( fabs( tmp[6].z.r - tmp[10].z.r ) < RT_EPSILON ) ) {

        return_value = 1;

    }

    cplex_vec_copy( icept_pt, tmp+6);

    printf("\n--------------------------------------------\n");

    return ( return_value );

}

