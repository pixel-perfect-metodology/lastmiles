
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
 *
 * See v.h for MATH_OP_SUCCESS and MATH_OP_FAIL
 */

int line_plane_icept( vec_type *icept_pt,
                      vec_type *plun,
                      vec_type *plvn,
                      vec_type *kst,
                      vec_type *lp0, vec_type *lpr,
                      vec_type *pl0, vec_type *pn,
                      vec_type *plu, vec_type *plv) 
{
    int line_in_plane = 0;
    int return_value = 0;
    cplex_type ctmp[12];
    vec_type i_hat, j_hat, lpr_norm, pn_norm,
             pl0_lp0_dir, pl0_lp0_dirn, tmp[15];


    /* be careful of uninitialized memory on the stack */
    memset( &ctmp, 0x00, (size_t)(12)*sizeof(cplex_type));
    memset( &tmp, 0x00, (size_t)(15)*sizeof(vec_type));

    double lpr_pn_theta, u_mag, v_mag;

    /* vars we may need for a line plane minimal distance
     * calculation */
    double min_dist_e, hypoteneuse, base_length;

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
        return MATH_OP_FAIL;
    }

    /* check up front that we are not dealing with zero
     * magnitude vectors */
    if (    ( fabs(lpr->x.r) < RT_EPSILON ) 
         && ( fabs(lpr->y.r) < RT_EPSILON )  
         && ( fabs(lpr->z.r) < RT_EPSILON )  ) {

        fprintf(stderr,"FAIL : line direction vector too small\n");

        if ( ( lpr->x.r == 0.0 ) && ( lpr->y.r == 0.0 ) && ( lpr->z.r == 0.0 ) ) {
            fprintf(stderr,"     : in fact it is zero magnitude\n");
        }
        return MATH_OP_FAIL;
    }

    if (    ( fabs(pn->x.r) < RT_EPSILON ) 
         && ( fabs(pn->y.r) < RT_EPSILON )  
         && ( fabs(pn->z.r) < RT_EPSILON )  ) {

        fprintf(stderr,"FAIL : plane normal vector too small\n");

        if ( ( pn->x.r == 0.0 ) && ( pn->y.r == 0.0 ) && ( pn->z.r == 0.0 ) ) {
            fprintf(stderr,"     : in fact it is zero magnitude\n");
        }
        return MATH_OP_FAIL;
    }

    if ( cplex_vec_mag( lpr ) < RT_EPSILON ) {
        fprintf(stderr,"FAIL : line direction vector too small\n");
        return MATH_OP_FAIL;
    }

    if ( cplex_vec_mag( pn ) < RT_EPSILON ) {
        fprintf(stderr,"FAIL : plane normal vector too small\n");
        return MATH_OP_FAIL;
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
     * of intercepts however the "reasonable" solution is simply the point
     * on the line that is closest to the plane point pl0.
     *
     * To be more clear we could argue that a line in a plane has
     * an infinite number of intercept solutions and that if one
     * were to select a single intercept point then it "should" be
     * the point on the line which is nearest to the provided
     * plane point pl0. While that may make sense from a geometric
     * perspective it does not help with ray tracing.
     */ 

    /* we will also need the i and j basis vectors */
    cplex_vec_set ( &i_hat, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    cplex_vec_set ( &j_hat, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);

    if ( cplex_vec_normalize( &lpr_norm, lpr ) == EXIT_FAILURE ) {
        fprintf(stderr,"FAIL : impossible to normalize vec lpr\n");
        return MATH_OP_FAIL;
    }

    if ( cplex_vec_normalize( &pn_norm, pn ) == EXIT_FAILURE ) {
        fprintf(stderr,"FAIL : impossible to normalize vec pn\n");
        return MATH_OP_FAIL;
    }

    /* we will need a direction vector from the plane point pl0 to the
     * line point lp0 below. We create this vector in pl0_lp0_dir. */
    pl0_lp0_dir.x.r = lp0->x.r - pl0->x.r;
    pl0_lp0_dir.x.i = lp0->x.i - pl0->x.i;
    pl0_lp0_dir.y.r = lp0->y.r - pl0->y.r;
    pl0_lp0_dir.y.i = lp0->y.i - pl0->y.i;
    pl0_lp0_dir.z.r = lp0->z.r - pl0->z.r;
    pl0_lp0_dir.z.i = lp0->z.i - pl0->z.i;

    /* check if we have a zero magnitude here on pl0_lp0_dir */
    if ( ( fabs(pl0_lp0_dir.x.r) < RT_EPSILON ) &&
         ( fabs(pl0_lp0_dir.y.r) < RT_EPSILON ) &&
         ( fabs(pl0_lp0_dir.z.r) < RT_EPSILON ) ) {

        fprintf(stderr,"WARN : plane point and line point are");
        if ( ( pl0_lp0_dir.x.r == 0.0 ) &&
             ( pl0_lp0_dir.y.r == 0.0 ) &&
             ( pl0_lp0_dir.z.r == 0.0 ) ) {
            fprintf(stderr,"the exact same point\n");
        } else {
            fprintf(stderr,"nearly the same\n");
        }

        /* degenerate trap here, we have KST == zero everywhere */
        kst->x.r = 0.0; kst->x.i = 0.0;
        kst->y.r = 0.0; kst->y.i = 0.0;
        kst->z.r = 0.0; kst->z.i = 0.0;

    } else {

        /* if we made it past the above sanity checks then
         * we normalize that into pl0_lp0_dirn */
        cplex_vec_normalize( &pl0_lp0_dirn, &pl0_lp0_dir );
    
        if ( cplex_vec_dot( ctmp, &lpr_norm, &pn_norm) == EXIT_FAILURE ) {
            /* The only way this can happen is if the dot product
             * result has a imaginary component. */
            fprintf(stderr,"FAIL : dot(lpr_norm, pn_norm) returned complex\n");
            return MATH_OP_FAIL;
        }
    
        /* Since the dot product of two normalized vectors results in the
         * cosine of the angle between them we can just check for a zero
         * result.  We know that any angle theta = pi/2 + 2 * n * pi will
         * be orthogonal for all integer n and cosine(theta) is zero in
         * such cases. We shall check for an orthogonal condition to
         * within a millionth of a degree with RT_ANGLE_COS_EPSILON */
    
        if ( fabs(ctmp[0].r) < RT_ANGLE_COS_EPSILON ) {
    
            fprintf(stderr,"WARN : lpr and pn are ");
            if ( fabs(ctmp[0].r) == 0.0 ) {
                fprintf(stderr,"perfectly");
            } else {
                fprintf(stderr,"nearly");
            }
            fprintf(stderr," orthogonal\n");
    
            lpr_pn_theta = acos(ctmp[0].r);
    
            fprintf(stderr,"INFO : lpr and pn angle = %-+18.12e radians\n",
                                           lpr_pn_theta );
    
            fprintf(stderr,"INFO : which is %-+18.12e degrees\n",
                                           lpr_pn_theta * 180.0 / M_PI );
    
            /* Since the line is nearly ( or perfectly ) orthogonal to the
             * plane normal we need to check if the line is actually in
             * the plane. Here we check if the direction vector from the
             * plane point pl0 to the line point lp0 is also orthogonal to
             * the plane normal. If so then the point lp0 is in the plane
             * and that clearly means that the entire line is in the plane.
             *
             * NOTE :     w a r n i n g 
             *
             *          This check will not work if the line point lp0
             *          and the plane point pl0 are both the same. However
             *          we trapped for this condition above.
             */
            if ( cplex_vec_dot( ctmp, &pn_norm, &pl0_lp0_dirn ) == EXIT_FAILURE ) {
                fprintf(stderr,"FAIL : dot(pn_norm, pl0_lp0_dirn) returned complex\n");
                return MATH_OP_FAIL;
            }
    
            if ( fabs(ctmp[0].r) < RT_ANGLE_COS_EPSILON ) { 
                fprintf(stderr,"WARN : line is in the plane\n");
    
                /* This really is a non-issue. We have an infinite number
                 * of intercept points to choose from and we shall deal
                 * with this below.  */
                line_in_plane = 1;
    
            } else {
                /* The line is perfectly orthogonal to the plane normal
                 * with no possible intercepts. The line is parallel to 
                 * the plane in this case. */
                fprintf(stderr,"FAIL : no possible intercept\n");
                return MATH_OP_FAIL;
            }
        }
    }

    /* If plu and plv both exist then we need to check that
     * they are linearly independant and then create the 
     * normalized versions of them. If they do not exist then
     * we have the task of creation based on the existing basis
     * vectors i_hat and j_hat. We may also have the situation
     * where only one of them exists and we must compute the
     * other.
     *
     * Even worse we need to verify that the supplied u and v
     * vectors are actually in the plane and they are orthogonal.
     *
     * NOTE : the restriction on orthogonal basic vectors u and v
     *        will mean we may never use a skew system in the 
     *        plane.
     */

    if ( ( plu == NULL ) && ( plv == NULL ) ) {
        /* We must compute both the plu and plv where the
         * i_hat or j_hat basis vectors are used.
         *
         * This is used as an entry point if we need to
         * reject the supplied u and v vectors. */
uv:     if ( cplex_vec_dot( ctmp+1, &pn_norm, &i_hat) == EXIT_FAILURE ) {
            fprintf(stderr,"FAIL : dot(pn_norm, i_hat) returned complex\n");
            return MATH_OP_FAIL;
        }

        /* if cosine(theta) is equal to either positive one or
         * negative one then we have perfect linear alignment
         * between the plane normal and i_hat. Thus we move on
         * and test with j_hat basis vector. */
        if ( fabs(fabs(ctmp[1].r) - 1.0) < RT_EPSILON ) {
            /* we need to use j_hat instead */
            if ( cplex_vec_dot( ctmp+1, &pn_norm, &j_hat) == EXIT_FAILURE ) {
                fprintf(stderr,"FAIL : dot(pn_norm, j_hat) returned complex\n");
                return MATH_OP_FAIL;
            }

            /* we have sufficient angle between i_hat and the plane
             * normal to continue. */
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
                /* check that plv is zero magnitude */
                if ( ( fabs(plv->x.r) < RT_EPSILON ) &&
                     ( fabs(plv->y.r) < RT_EPSILON ) &&
                     ( fabs(plv->z.r) < RT_EPSILON ) ) {
                    /* we have a plv vector that is too small to
                     * work with and thus we need to generate u and v
                     * ourselves */
                    goto uv;
                }

                /* check if plv is actually in the plane */
                cplex_vec_normalize( plvn, plv );
                if ( cplex_vec_dot( ctmp, &pn_norm, plvn ) == EXIT_FAILURE ) {
                    fprintf(stderr,"FAIL : dot(pn_norm, plvn) returned complex\n");
                    return MATH_OP_FAIL;
                }

                /* If plv is orthogonal to the plane normal then the
                 * cosine of the angle will be zero. */
                if ( fabs(ctmp->r) > RT_EPSILON ) {
                    fprintf(stderr,"WARN : v vector not in the plane.\n");
                    goto uv;
                }

                /* compute plu */
                cplex_vec_cross( tmp, &pn_norm, plvn );
                cplex_vec_normalize( plun, tmp );
            } else {
                /* check that plu is zero magnitude */
                if ( ( fabs(plu->x.r) < RT_EPSILON ) &&
                     ( fabs(plu->y.r) < RT_EPSILON ) &&
                     ( fabs(plu->z.r) < RT_EPSILON ) ) {
                    goto uv;
                }

                /* check if plu is actually in the plane */
                cplex_vec_normalize( plun, plu );
                if ( cplex_vec_dot( ctmp, &pn_norm, plun ) == EXIT_FAILURE ) {
                    fprintf(stderr,"FAIL : dot(pn_norm, plun) returned complex\n");
                    return MATH_OP_FAIL;
                }

                if ( cplex_vec_dot( ctmp, &pn_norm, plun ) == EXIT_FAILURE ) {
                    fprintf(stderr,"FAIL : dot(pn_norm, plun) returned complex\n");
                    return MATH_OP_FAIL;
                }

                /* If plu is orthogonal to the plane normal then 
                 * the cosine of the angle will be zero. */
                if ( fabs(ctmp->r) > RT_EPSILON ) {
                    fprintf(stderr,"WARN : u vector not in the plane.\n");
                    goto uv;
                }

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
                            return MATH_OP_FAIL;

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
                if ( cplex_vec_dot( ctmp, plv, plu ) == EXIT_FAILURE ) {
                    return return_value;
                }

                if ( fabs(fabs( ctmp->r ) - 1.0) < RT_EPSILON ) {
                    /* the u and v vectors are so close to linear that
                     * we shall call them useless */
                    fprintf(stderr,"WARN : u and v are near linear.\n");
                    goto uv;
                }
            }
        }
    }

    printf("dbug : end of uv vector computation %s:%d\n",
                                           __FILE__, __LINE__);

    printf("     : u_hat = %+-16.9e", plun->x.r);
    printf("    %+-16.9e", plun->y.r);
    printf("    %+-16.9e\n", plun->z.r );
    printf("     : v_hat = %+-16.9e", plvn->x.r);
    printf("    %+-16.9e", plvn->y.r);
    printf("    %+-16.9e\n", plvn->z.r );

    if ( line_in_plane ) {
        fprintf(stderr,"WARN : line is in the plane\n");
        /* degenerate trap here, we have KST == zero everywhere */
        kst->x.r = 0.0; kst->x.i = 0.0;
        kst->y.r = 0.0; kst->y.i = 0.0;
        kst->z.r = 0.0; kst->z.i = 0.0;

        /* What do we know at this point? 
         *
         * we have : lpr_norm = line direction normalized 
         *            pn_norm = plan normal and yes it is normalized
         *
         *                lp0 = the point given to us in the plane
         *                pl0 = the point given to us on the line
         *
         *        pl0_lp0_dir = vector Lp0 - Pl0 = vector g in diagram
         */

        printf("dbug : pl0_lp0_dir = %+-16.9e", pl0_lp0_dir.x.r);
        printf("    %+-16.9e", pl0_lp0_dir.y.r);
        printf("    %+-16.9e\n", pl0_lp0_dir.z.r);
        printf("     : this is Lp0 - Pl0 which is g vector in diagram\n\n");

        printf("dbug : lpr_norm = %+-16.9e", lpr_norm.x.r);
        printf("    %+-16.9e", lpr_norm.y.r);
        printf("    %+-16.9e\n", lpr_norm.z.r);
        printf("     : this is n_hat\n\n");

        if ( cplex_vec_dot( ctmp, &pl0_lp0_dir, &lpr_norm ) == EXIT_FAILURE ) {
            fprintf(stderr,"FAIL : dot(pl0_lp0_dir,lpr_norm) returned complex\n");
            return return_value;
        }
        printf("dbug : g dot n = %+-16.9e\n", ctmp[0].r );

        /* scale lpr_norm by the factor from the dot product */
        cplex_vec_scale( tmp+11, &lpr_norm, ctmp[0].r);

        printf("dbug : (g dot n) * n_hat = %+-16.9e", tmp[11].x.r);
        printf("    %+-16.9e", tmp[11].y.r);
        printf("    %+-16.9e\n", tmp[11].z.r );

        /* okay negate that and then add to ( Lp0 - Pl0 ) */
        tmp[13].x.r = pl0_lp0_dir.x.r - tmp[11].x.r;
        tmp[13].y.r = pl0_lp0_dir.y.r - tmp[11].y.r;
        tmp[13].z.r = pl0_lp0_dir.z.r - tmp[11].z.r;

        printf("dbug : g - ( (g dot n) * n_hat ) = %+-16.9e", tmp[13].x.r);
        printf("    %+-16.9e", tmp[13].y.r);
        printf("    %+-16.9e\n\n", tmp[13].z.r );

        min_dist_e = cplex_vec_mag(tmp+13);
        printf("dbug : minimal distance to line should be %+-16.9e\n\n",
                          min_dist_e );

        hypoteneuse = sqrt( ( pl0_lp0_dir.x.r * pl0_lp0_dir.x.r )
                          + ( pl0_lp0_dir.y.r * pl0_lp0_dir.y.r )
                          + ( pl0_lp0_dir.z.r * pl0_lp0_dir.z.r ) );

        printf("dbug : hypoteneuse = %+-16.9e\n", hypoteneuse );

        base_length = sqrt ( ( hypoteneuse * hypoteneuse ) 
                           - ( min_dist_e * min_dist_e ) );

        printf("dbug : base_length = %+-16.9e\n", base_length );

        /* the intercept point on the line that is nearest to the plane
         * point should be 
         *
         *      J = base_length * lpr_norm + lp0
         *
         * we shall use tmp[14] for the point J for now
         */

        tmp[14].x.r = base_length * lpr_norm.x.r + lp0->x.r;
        tmp[14].y.r = base_length * lpr_norm.y.r + lp0->y.r;
        tmp[14].z.r = base_length * lpr_norm.z.r + lp0->z.r;

        printf("dbug : intercept J at < %+-16.9e, %+-16.9e, %+-16.9e >\n",
                tmp[14].x.r, tmp[14].y.r, tmp[14].z.r );

        /* save the result back into the supplied vec_type pointer */
        cplex_vec_copy( icept_pt, tmp+14);

        return MATH_OP_SUCCESS;

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

    cplex_det( ctmp+2, &v[0], &v[1], &v[2] ); 
    printf("     :   det =    %+-16.9e + %+-16.9e i\n", ctmp[2].r, ctmp[2].i);

    /* check if the determinant is so very close to zero that it
     * may be merely noise. In fact check if it is smaller than
     * the square of RT_EPSILON */
    if ( ( fabs(ctmp[2].r) < ( RT_EPSILON * RT_EPSILON ) ) 
            && ( ctmp[2].r != 0.0 ) ) {
        printf("     :   det is microscopic noise\n");
        ctmp[2].r = 0.0;
        ctmp[2].i = 0.0;
        printf("     :   det dropped to zero\n");
    }

    printf("\nSolve for line plane intercept with Cramers rule.\n\n");
    if ( cplex_cramer(&res_vec, &v[0], &v[1], &v[2], &rh_col) != 0 ) {
        fprintf(stderr,"dbug : cplex_cramer reports no valid solution\n");
        fprintf(stderr,"     : %s at %d\n", __FILE__, __LINE__ );
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

    fprintf(stderr,"dbug : kst->x.r == k is %-+16.9e\n", kst->x.r);
    fprintf(stderr,"     : %s at %d\n", __FILE__, __LINE__ );
    /* check if k is very much smaller than RT_EPSILON */
    if ( ( kst->x.r != 0.0 ) && ( fabs(kst->x.r) < ( RT_EPSILON * RT_EPSILON ) ) ) {
        fprintf(stderr,"     : kst->x.r == k is microscopic noise\n");
        kst->x.r = 0.0;
        fprintf(stderr,"     : kst->x.r == k has been dropped to zero\n");
    }
    /* multiply     k * norm[ lpr ]     */
    cplex_vec_scale( tmp+5, &lpr_norm, kst->x.r);
    cplex_vec_add( tmp+6, lp0, tmp+5 );


    /* Since tmp[6] is used as the final result to be returned as
     * the intercept point and thus it can be slightly cleaned
     * of noisey very very small values.
     *
     * Check for values much less than RT_EPSILON and cull them
     * to zero. */
    if ( ( fabs(tmp[6].x.r) < RT_EPSILON ) ||
         ( fabs(tmp[6].y.r) < RT_EPSILON ) ||
         ( fabs(tmp[6].z.r) < RT_EPSILON ) ) {
        fprintf(stderr,"dbug : cull below RT_EPSILON values\n");
        fprintf(stderr,"     : %s at %d\n", __FILE__, __LINE__ );
        if ( fabs(tmp[6].x.r) < RT_EPSILON ) {
            fprintf(stderr,"     : x dropped from %-+16.9e to zero\n", tmp[6].x.r);
            tmp[6].x.r = 0.0;
        }
        if ( fabs(tmp[6].y.r) < RT_EPSILON ) {
            fprintf(stderr,"     : y dropped from %-+16.9e to zero\n", tmp[6].y.r);
            tmp[6].y.r = 0.0;
        }
        if ( fabs(tmp[6].z.r) < RT_EPSILON ) {
            fprintf(stderr,"     : z dropped from %-+16.9e to zero\n", tmp[6].z.r);
            tmp[6].z.r = 0.0;
        }
    }

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

        return_value = MATH_OP_SUCCESS;

    }

    cplex_vec_copy( icept_pt, tmp+6);

    return return_value;

}

