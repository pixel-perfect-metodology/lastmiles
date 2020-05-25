
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
    int status, line_in_plane, return_value = 0;
    cplex_type ctmp[12];
    vec_type i_hat, j_hat, lpr_norm, pn_norm,
             pl0_lp0_dir, pl0_lp0_dirn, tmp[12];
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
       ||( pn  == NULL ) ) {
        return return_value;
    }

    /* need vectors of a reasonable length to work with */
    if (  ( cplex_vec_mag( lpr ) < RT_EPSILON ) 
       || ( cplex_vec_mag( pn  ) < RT_EPSILON ) ) {
        return return_value;
    }

    /* There are a few degenerate cases to check for. Firstly the
     * point on the line lp0 may be the same as the point in the
     * plane pl0. Unlikely but it solves the entire intercept
     * computation right away. What would remain is the need for
     * reasonable plane_u and plane_v vectors. However we do have 
     * k = s = t = 0 in this degenerate case.
     *
     * Another possible situation is that the line may actually be
     * in the plane. In such a situation there are an infinite number
     * of intercepts however the reasonable solution is simply the point
     * on the line that is closest to the plane point pl0. This gives
     * us value for k wherein the line and point pl0 are closest.
     *
     * need to verify that the vectors u and v are sane. However some
     * minimal non-zero k will certainly result in non-zero s and t.
     *
     * To be more clear we could argue that a line in a plane has
     * an infinite number of intercept solutions and that if one
     * were to select a single intercept point then it "should" be
     * the point on the line which is nearest to the provided
     * plane point pl0. While that may make sense from a geometric
     * perspective it does not help with ray tracing. We therefore
     * merely accept that the line parameter k is zero and thus the
     * intercept is the point lp0 provided for the line and we 
     *
     *
     *
     * */ 

    /* we will need a direction vector from the plane point pl0 to the
     * line point lp0 below. We create this vector in pl0_lp0_dir. */
    pl0_lp0_dir.x.r = lp0->x.r - pl0->x.r;
    pl0_lp0_dir.x.i = lp0->x.i - pl0->x.i;
    pl0_lp0_dir.y.r = lp0->y.r - pl0->y.r;
    pl0_lp0_dir.y.i = lp0->y.i - pl0->y.i;
    pl0_lp0_dir.z.r = lp0->z.r - pl0->z.r;
    pl0_lp0_dir.z.i = lp0->z.i - pl0->z.i;
    /* normalize that into pl0_lp0_dirn */
    cplex_vec_normalize( &pl0_lp0_dirn, &pl0_lp0_dir );

    /* we will also need the i and j basis vectors */
    cplex_vec_set ( &i_hat, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    cplex_vec_set ( &j_hat, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);

    /* check if the line and the plane normal are orthogonal */
    cplex_vec_normalize( &lpr_norm, lpr );

    cplex_vec_normalize( &pn_norm, pn );

    cplex_vec_dot( ctmp, &lpr_norm, &pn_norm);

    if ( check_dot( ctmp ) == EXIT_FAILURE )
        return return_value;

    /* Since the dot product of two normalized vectors results in the
     * cosine of the angle between them we can just check for a zero
     * result.  We know that any angle theta = pi/2 + 2 * n * pi will
     * be orthogonal for all integer n and cosine(theta) is zero in
     * such cases. */

    line_in_plane = 0;
    if ( fabs(ctmp[0].r) < RT_ANGLE_COS_EPSILON ) {

        /*
        fprintf(stderr,"WARN : lpr and pn are orthogonal.\n");
        */

        /* Since the line is perfectly orthogonal to the plane normal
         * we need to check if the line is actually in the plane. Here
         * we check if the direction vector from the plane point pl0 to
         * the line point lp0 is also orthogonal to the plane normal.
         *
         * If so then the point lp0 is in the plane and that clearly
         * means that the entire line is in the plane. */

        /* dot product of pl0_lp0_dirn and the plane normal */
        cplex_vec_dot( ctmp, &pn_norm, &pl0_lp0_dirn );

        if ( fabs(ctmp[0].r) < RT_ANGLE_COS_EPSILON ) { 

            fprintf(stderr,"WARN : line is in the plane\n");

            /* This really is a non-issue. We have an infinite number
             * of intercept points to choose from and we shall deal
             * with this below.
             *
             * Long overdue is to deal with this. 
             *
             * Shall we simple take the point nearest to the plane
             * point pl0 ?
             */

            line_in_plane = 1;

        } else {
            /* This really is an impossible situation. The line is
             * perfectly orthogonal to the plane normal with no
             * possible intercepts. */
            fprintf(stderr,"FAIL : no possible lp intercept\n");
            return return_value;
        }
    }

    /* If plu and plv both exist then we need to check that
     * they are linearly independant and then create the 
     * normalized versions of them. If they do not exist then
     * we have the task of creation based on the existing basis
     * vectors i_hat and j_hat. We may also have the situation
     * where only one of them exists and we must compute the
     * other.
     */

    if ( ( plu == NULL ) && ( plv == NULL ) ) {
        /* We must compute both the plu and plv where the
         * i_hat or j_hat basis vectors are used. */

uv:     cplex_vec_dot( ctmp+1, &pn_norm, &i_hat);

        if ( check_dot( ctmp+1 ) == EXIT_FAILURE )
            return return_value;

        /* if cosine(theta) is equal to either positive one or
         * negative one then we have perfect linear alignment
         * between the plane normal and i_hat. Thus we move on
         * and test with j_hat basis vector. */
        if ( fabs(fabs(ctmp[1].r) - 1.0) < RT_EPSILON ) {
            /* we need to use j_hat instead */
            cplex_vec_dot( ctmp+1, &pn_norm, &j_hat);
            if ( check_dot( ctmp+1 ) == EXIT_FAILURE )
                return return_value;

            cplex_vec_copy( tmp+2, &j_hat);

        } else {

            /* use i_hat as the reference basis vector */
            cplex_vec_copy( tmp+2, &i_hat);
        }

        /* create an orthogonal vector plu in tmp+3 */
        cplex_vec_cross( tmp+3, &pn_norm, tmp+2 );

        /* normalize that into plun */
        cplex_vec_normalize( plun, tmp+3 );

        /* We have plun and may now proceed to create an orthogonal
         * vector plv in tmp+4 */
        cplex_vec_cross( tmp+4, &pn_norm, plun );
        cplex_vec_normalize( plvn, tmp+4 );

    } else {
        /* We know that we have at least plu or plv now */
        if ( ( plu == NULL ) || ( plv == NULL ) ) {
            /* we only have one of them */
            if ( plu == NULL ) {
                /* check if plv is actually in the plane */
                if ( cplex_vec_mag( plv ) < RT_EPSILON )
                    return return_value;

                cplex_vec_normalize( plvn, plv );
                cplex_vec_dot( ctmp, &pn_norm, plvn );

                if ( check_dot( ctmp ) == EXIT_FAILURE )
                    return return_value;

                /* If plv is orthogonal to the plane normal then the
                 * cosine of the angle will be zero. */
                if ( fabs(ctmp->r) > RT_EPSILON )
                    return return_value;

                /* compute plu */
                cplex_vec_cross( tmp, &pn_norm, plvn );
                cplex_vec_normalize( plun, tmp );
            } else {
                /* check if plu is actually in the plane */
                if ( cplex_vec_mag( plu ) < RT_EPSILON )
                    return return_value;

                cplex_vec_normalize( plun, plu );
                cplex_vec_dot( ctmp, &pn_norm, plun );

                if ( check_dot( ctmp ) == EXIT_FAILURE )
                    return return_value;

                /* If plu is orthogonal to the plane normal then 
                 * the cosine of the angle will be zero. */
                if ( fabs(ctmp->r) > RT_EPSILON )
                    return return_value;

                /* compute plv */
                cplex_vec_cross( tmp, &pn_norm, plun );
                cplex_vec_normalize( plvn, tmp );
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

            if ( ( u_mag < RT_EPSILON ) || ( v_mag < RT_EPSILON ) ) {

                /* One or both are very small. Is either zero ? */
                if ( ( u_mag == 0.0 ) || ( v_mag == 0.0 ) ) {
                    /* Well one of them is zero magnitude.
                     * Do we have a u vector ? */

                    if ( u_mag == 0.0 ) {

                        /* u vector is zero so lets ask about v and
                         * see if it is reasonable. Otherwise just
                         * compute them both as above. */
                        if (( v_mag == 0.0 ) || ( v_mag < RT_EPSILON ))
                            goto uv;

                        /* normalize plv */
                        cplex_vec_normalize( plvn, plv );

                        /* check if plvn is orthogonal to pn and
                         * if not then start over. Bear in mind that
                         * we have no idea if pn is normalized and 
                         * thus we must use tmp[1] from above which is
                         * the plane normal actually normalized. */

                        cplex_vec_dot( ctmp, &pn_norm, plvn );
                        if ( check_dot( ctmp ) == EXIT_FAILURE )
                            return return_value;

                        if ( ctmp->r != 0.0 ) goto uv;

                        /* we have a valid plvn and may compute plu with
                         * a cross product of pn and plvn */
                        cplex_vec_cross( tmp+3, pn, plvn );
                        /* normalize that into plun */
                        cplex_vec_normalize( plun, tmp+3 );

                    } else if ( v_mag == 0.0 ) {

                        /* v vector is zero. same situation as above
                         * we need to check u vector and then decide
                         * if we need to abandon the compute here and
                         * merely re-compute both u and v vectors. */
                        if (( u_mag == 0.0 ) || ( u_mag < RT_EPSILON ))
                            goto uv;

                        /* normalize plu  */
                        cplex_vec_normalize( plun, plu );

                        /* check if plun is orthogonal to pn. */
                        cplex_vec_dot( ctmp, &pn_norm, plun );
                        if ( check_dot( ctmp ) == EXIT_FAILURE )
                            return return_value;

                        if ( ctmp->r != 0.0 ) goto uv;

                        /* we have a valid plun and may compute plv
                         * with a cross product of pn and plun */
                        cplex_vec_cross( tmp+3, pn, plun );
                        /* normalize that into plvn */
                        cplex_vec_normalize( plvn, tmp+3 );

                    } else {

                        /* Both u and v are zero magnitude. So
                         * just start over and compute them both. */
                        goto uv;

                    }
                } else {
                    /* Neither u nor v is zero in size but at least
                     * one of them is smaller than RT_EPSILON. */
                    if ( u_mag < RT_EPSILON ) {
                        /* check plv and the first step is
                         * to normalize it */
                        cplex_vec_normalize( plvn, plv );

                        /* check if plvn is orthogonal to pn. */
                        cplex_vec_dot( ctmp, &pn_norm, plvn );
                        if ( check_dot( ctmp ) == EXIT_FAILURE )
                            return return_value;

                        if ( fabs(ctmp->r) > RT_EPSILON )
                            goto uv;

                        /* compute plu with a cross product of pn and plvn */
                        cplex_vec_cross( tmp+3, pn, plvn );
                        /* normalize that into plun */
                        cplex_vec_normalize( plun, tmp+3 );

                    } else {
                        /* here we know that v_mag < RT_EPSILON */
                        cplex_vec_normalize( plun, plu );
                        cplex_vec_dot( ctmp, &pn_norm, plun );
                        if ( check_dot( ctmp ) == EXIT_FAILURE )
                            return return_value;

                        if ( fabs(ctmp->r) > RT_EPSILON )
                            goto uv;

                        cplex_vec_cross( tmp+3, pn, plun );
                        cplex_vec_normalize( plvn, tmp+3 );
                    }
                }
            } else {
                /* we have both u and v vectors and they have some 
                 * reasonable magnitude. We have no idea if they are
                 * in the plane or even if they are linearly dependant
                 * or not. We hope not. In fact we need "not".
                 *
                 * The first task should be to determine that they 
                 * are linearly independant and thus the dot product
                 * of u and v will NOT be 1 or -1. */
                cplex_vec_dot( ctmp, plv, plu );
                if ( check_dot( ctmp ) == EXIT_FAILURE )
                    return return_value;

                if ( fabs(fabs( ctmp->r ) - 1.0) < RT_EPSILON ) {
                    /* the u and v vectors are so close to linear that
                     * we shall call them useless */
                    fprintf(stderr,"WARN : u and v are linear.\n");
                    goto uv;
                }
            }
        }
    }

    /*
    printf("dbug : u_hat = %+-16.9e", plun->x.r);
    printf("    %+-16.9e", plun->y.r);
    printf("    %+-16.9e\n", plun->z.r );
    printf("dbug : v_hat = %+-16.9e", plvn->x.r);
    printf("    %+-16.9e", plvn->y.r);
    printf("    %+-16.9e\n", plvn->z.r );
    */

    if ( line_in_plane ) {
        /* we have the line in the plane and thus the parameter
         * k for the line equation is zero. We are left to determine
         * the scalar parameters for s and t with respect to the
         * plane basis vectors plun and plvn. */
        printf("INFO : the line is in the plane. k is zero.\n");
        /* this would set the kst vector component for k to zero
         *
         *     kst->x.r = 0.0;
         *     kst->x.i = 0.0;
         */

        /* TODO : sort this shit out. This is not a valid 
         *        response to the conditions. We need to compute
         *        the nearest point on the line and the plane
         *        point pl0 */





        return return_value;

    }

    /* lets create the column of data for P3 - P0 in our
     * diagram. This would be  pl0 - lp0. */
    cplex_vec_set ( &rh_col, pl0->x.r - lp0->x.r,
                             pl0->x.i - lp0->x.i,
                             pl0->y.r - lp0->y.r,
                             pl0->y.i - lp0->y.i,
                             pl0->z.r - lp0->z.r,
                             pl0->z.i - lp0->z.i );

    v[0].x.r = lpr_norm.x.r;      v[0].x.i = lpr_norm.x.i;
    v[0].y.r = plun->x.r;         v[0].y.i = plun->x.i;
    v[0].z.r = plvn->x.r;         v[0].z.i = plvn->x.i;

    v[1].x.r = lpr_norm.y.r;      v[1].x.i = lpr_norm.y.i;
    v[1].y.r = plun->y.r;         v[1].y.i = plun->y.i;
    v[1].z.r = plvn->y.r;         v[1].z.i = plvn->y.i;

    v[2].x.r = lpr_norm.z.r;      v[2].x.i = lpr_norm.z.i;
    v[2].y.r = plun->z.r;         v[2].y.i = plun->z.i;
    v[2].z.r = plvn->z.r;         v[2].z.i = plvn->z.i;

    /*
    printf("Matrix with line plane intercept data.\n");
    printf("dbug : row 1 =    %+-16.9e    %+-16.9e    %+-16.9e\n",
            v[0].x.r, v[0].y.r, v[0].z.r );

    printf("     : row 2 =    %+-16.9e    %+-16.9e    %+-16.9e\n",
            v[1].x.r, v[1].y.r, v[1].z.r );

    printf("     : row 3 =    %+-16.9e    %+-16.9e    %+-16.9e\n",
            v[2].x.r, v[2].y.r, v[2].z.r );

    printf("dbug : rh_col= ");
    printf("%+-16.9e    %+-16.9e    %+-16.9e\n\n",
                 rh_col.x.r, rh_col.y.r, rh_col.z.r);

    printf("     :   det =    %+-16.9e\n", ctmp[2].r);

    printf("\nSolve for line plane intercept with Cramers rule.\n\n");
    */

    cplex_det( ctmp+2, &v[0], &v[1], &v[2] ); 
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
     * other.
     */

    /* multiply     k * norm[ lpr ]     */
    cplex_vec_scale( tmp+5, &lpr_norm, kst->x.r);
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

    return return_value;

}

