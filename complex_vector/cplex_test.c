
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
#include <complex.h>

#include "v.h"

int main ( int argc, char **argv)
{

    cplex_type op1, op2, op3, opr, opr2[3], quad_res[4];

    /* rh_col is right hand column for Cramer call with
     * res_vec as the result if it exists */
    vec_type v[4], rh_col, res_vec;

    /* elements for the line plane intercept test */
    vec_type line_point, line_direction, plane_point, plane_normal;
    vec_type lp_intercept_point;
    vec_type plane_u, plane_v, plane_u_norm, plane_v_norm;
    vec_type lp_intercept_param;

    int real_root_count;

    op1.i = 1.0; op1.r = 0.0;
    op2.i = 1.0; op2.r = 0.0;
    printf("dbug : op1 = ( %g, %g )\n", op1.r, op1.i);
    printf("     : op2 = ( %g, %g )\n", op2.r, op2.i);
    cplex_mult(&opr, &op1, &op2);
    printf("     : opr = op1 * op2 = ( %g, %g )\n", opr.r, opr.i);
    printf("     :     should be i^2 = -1\n\n");


    op1.r = 1.0;
    op1.i = 0.0;
    printf("dbug : op1 = ( %g, %g )\n", op1.r, op1.i);
    printf("     : op2 = ( %g, %g )\n", op2.r, op2.i);
    cplex_mult(&opr, &op1, &op2);

    printf("     : opr = op1 * op2 = ( %g, %g )\n", opr.r, opr.i);
    printf("     :     should be 1i\n\n");

    op1.r = -1.0; op1.i = 0.0;
    op2.r = 0.0; op2.i = 0.0;
    printf("dbug : op1 = ( %g, %g )\n", op1.r, op1.i);
    printf("     : op2 = ( %g, %g )\n", op2.r, op2.i);
    cplex_mult(&opr, &op1, &op2);
    printf("     : opr = op1 * op2 = ( %g, %g )\n", opr.r, opr.i);
    printf("     :     should be just zero.\n\n");

    op1.r = 4.0;
    op1.i = 3.0;
    op2.r = 2.0;
    op2.i = -1.0;
    printf("dbug : op1 = ( %g, %g )\n", op1.r, op1.i);
    printf("     : op2 = ( %g, %g )\n", op2.r, op2.i);
    cplex_div(&opr, &op1, &op2);
    printf("dbug : opr = op1 / op2 = ( %g, %g )\n", opr.r, opr.i);
    printf("     :     should be 1 + 2i\n\n");

    op2.r = -4.0;
    op2.i = -12.0;
    printf("dbug : op1 = ( %g, %g )\n", op1.r, op1.i);
    printf("     : op2 = ( %g, %g )\n", op2.r, op2.i);
    cplex_div(&opr, &op1, &op2);
    printf("dbug : opr = op1 / op2 = ( %g, %g )\n", opr.r, opr.i);
    printf("     :         should be ( -0.325, 0.225 )\n\n");


    /* these next two angles can be confirmed with a trivial atan()
     *
     * a( 0.5 )
     * .46364760900080611621
     * a( 0.75 )
     * .64350110879328438680
     */
    op1.r = 2.0;
    op1.i = 1.0;
    printf("dbug : op1 = ( %g, %g )\n", op1.r, op1.i);
    printf("     :     theta = %16.12e\n", cplex_theta( &op1 ) );
    printf("     : should be   4.6364760900080611621e-01\n\n");


    op1.r = 4.0;
    op1.i = 3.0;
    printf("dbug : op1 = ( %g, %g )\n", op1.r, op1.i);
    printf("     :     theta = %16.12e\n", cplex_theta( &op1 ) );
    printf("     : should be   6.4350110879328438680e-01\n\n");

    /*
     * a(24/7)
     * 1.28700221758656877360
     */
    cplex_sq(&opr, &op1);
    printf("     : opr = op1^2 = ( %g, %g )\n", opr.r, opr.i);
    printf("     :     should be 7 + 24i\n");
    printf("     :     magnitude is %g\n", cplex_mag(&opr));
    printf("     :     theta = %16.12e\n", cplex_theta( &opr));
    printf("     : should be   1.28700221758656877360\n\n");


    printf("     : now we take the square root(s) of opr\n");
    op1.r = opr.r;
    op1.i = opr.i;
    cplex_sqrt(opr2, &op1);
    printf("root : 1 is ( %g, %g )\n", opr2[0].r, opr2[0].i);
    printf("root : 2 is ( %g, %g )\n", opr2[1].r, opr2[1].i);
    printf("     : should be ( 4, 3i ) and ( -4, -3i ).\n\n");

    /* square root of ( 0, 1 ) */
    printf("dbug : square root test\n");
    op1.r = 0.0;
    op1.i = 1.0;
    printf("dbug : op1 = ( %g, %g )\n", op1.r, op1.i);
    printf("     :     theta = %16.12e\n", cplex_theta(&op1) );
    printf("     :     magnitude is %g\n", cplex_mag(&op1));
    cplex_sqrt(opr2, &op1);
    printf("root : 1 = ( %16.12e, %16.12e )\n", opr2[0].r, opr2[0].i);
    printf("root : 2 = ( %16.12e, %16.12e )\n\n", opr2[1].r, opr2[1].i);


    /* cube roots of ( -11 + 2i )
     *
     *  sqrt(5) e^(1/3 i ( pi - tan^(-1)(2/11)))
     *  =  1.23205080756888 + 1.86602540378444 i (principal root)
     *
     *  sqrt(5) e^(1/3 i ( 3*pi - tan^(-1)(2/11)))
     *  = -2.23205080756888 + 0.133974596215561 i
     *
     *  sqrt(5) e^(i (-2 * pi + 1/3 (5*pi - tan^(-1)(2/11))))
     *  =  1.00000000000000 -2.00000000000000 i
     */
    printf("dbug : cube root test\n");
    op1.r = -11.0;
    op1.i = 2.0;
    printf("     : op1 = ( %g, %g )\n", op1.r, op1.i);
    printf("     :     theta = %16.12e\n", cplex_theta(&op1) );
    printf("     :     magnitude is %g\n", cplex_mag(&op1));

    cplex_cbrt(opr2, &op1);
    printf("root : 1 = ( %16.12e, %16.12e )\n", opr2[0].r, opr2[0].i);
    printf("           ( 1.23205080756888,   1.86602540378444 )\n");
    printf("root : 2 = ( %16.12e, %16.12e )\n", opr2[1].r, opr2[1].i);
    printf("           ( -2.23205080756888,   1.33974596215561e-01 )\n");
    printf("root : 3 = ( %16.12e, %16.12e )\n", opr2[2].r, opr2[2].i);
    printf("           (  1, -2 )\n\n");


    printf("\nTest the complex vector magnitude\n");
    /* Test with < (1 + 1i), (2 + 2i), (3+3i) >
     *
     * should be 2 * sqrt(7) = 5.291502622129181 */

    v[0].x.r = 1.0; v[0].x.i = 1.0;
    v[0].y.r = 2.0; v[0].y.i = 2.0;
    v[0].z.r = 3.0; v[0].z.i = 3.0;
    printf("dbug : v1 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[0].x.r, v[0].x.i,
            v[0].y.r, v[0].y.i,
            v[0].z.r, v[0].z.i );
    printf("     :    |v1| = %16.12e\n", cplex_vec_mag( v ));
    printf("     : Should be 5.2915026221291e+00\n\n");


    /* second vector is <( -1 - 1i), ( -2 -2i), ( 3 - 3i ) > */
    v[1].x.r = -1.0; v[1].x.i = -1.0;
    v[1].y.r = -2.0; v[1].y.i = -2.0;
    v[1].z.r =  3.0; v[1].z.i = -3.0;
    printf("     : v2 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[1].x.r, v[1].x.i,
            v[1].y.r, v[1].y.i,
            v[1].z.r, v[1].z.i );


    /* OKay not let us test the vector dot product of v[0] dot v[1]
     * where we currently have operands :
     *
     * < (1 + 1i), (2 + 2i), (3+3i) >
     *                    . <( -1 - 1i), ( -2 -2i), ( 3 - 3i ) >
     *
     * result should be ( 18, -10i ) */

    printf("Lets test vector dot product\n");
    cplex_vec_dot(&opr, v, v+1);
    printf("     : dot product = ( %g, %g )\n", opr.r, opr.i);
    printf("     : should be ( 18, -10i )\n\n");


    /*  Now let us test the vector cross product of v[0] X v[1]
     *  with the same operands as above :
     *
     *  < (1 + 1i), (2 + 2i), (3+3i) >
     *                    X <( -1 - 1i), ( -2 -2i), ( 3 - 3i ) >
     *
     * where again thankfully we have traviss on irc and tw0st3p
     * with julia to confirm output as the vector 
     *
     * < 12 + 12im, -6 - 6im, 0 + 0im > 
     */
    printf("Lets test vector cross product\n");
    cplex_vec_cross(v+2, v, v+1);
    printf("     : v1 X v2 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[2].x.r, v[2].x.i,
            v[2].y.r, v[2].y.i,
            v[2].z.r, v[2].z.i );



    /* Again thanks to cool data from Traviss we have some results
     * for a normalized vector v[0] thus :
     *
     * Thanks to Julia we see : 
     *
     * < ( 0.1889822365046136 + 0.1889822365046136im ),
     *   ( 0.3779644730092272 + 0.3779644730092272im ),
     *   ( 0.5669467095138407 + 0.5669467095138407im ) >
     */
    if ( cplex_vec_normalize( (v+3), v ) == MATH_OP_FAIL ) {
        printf("FAIL : cplex_vec_normalize() returns MATH_OP_FAIL\n");
        return EXIT_FAILURE;
    }

    printf("Lets test vector normalization of v[0] as described.\n");
    printf("    : |v[0]| = ");
    printf(" < ( %16.12e, %16.12e ),\n", v[3].x.r, v[3].x.i );
    printf("                  ( %16.12e, %16.12e ),\n", v[3].y.r, v[3].y.i );
    printf("                  ( %16.12e, %16.12e ) >\n", v[3].z.r, v[3].z.i );
    printf("      Should be < ( 0.1889822365046 + 0.1889822365046 i ),\n");
    printf("                  ( 0.3779644730092 + 0.3779644730092 i ),\n");
    printf("                  ( 0.5669467095138 + 0.5669467095138 i )>\n\n");


    /* We shall now test the solution to a complex quadratic polynomial.
     *
     * see https://en.wikipedia.org/wiki/Complex_quadratic_polynomial
     *
     * Begin with trivial x^2 - 9 * x + 14 = 0 which is just 
     *
     *     ( x - 2 ) * ( x - 7 ) = 0 
     *
     *     thus the roots are trivial real values 2 and 7.
     *
     */

    printf("Quadratic x^2 - 9 * x + 14 = 0\n");
    op1.r = 1.0; op1.i = 0.0;
    op2.r = -9.0; op2.i = 0.0;
    op3.r = 14.0; op3.i = 0.0;

    if ( cplex_quadratic( quad_res, &real_root_count,
                          &op1, &op2, &op3 ) == MATH_OP_FAIL ) {

        fprintf(stderr,"WARN : cplex_quadratic returns MATH_OP_FAIL\n");

    } else {

        printf("Real root count = %i\n", real_root_count );
        printf("Quadratic result 1 = ( %16.12e, %16.12e )\n",
                                          quad_res[0].r, quad_res[0].i);
        printf("          result 2 = ( %16.12e, %16.12e )\n\n",
                                          quad_res[1].r, quad_res[1].i);

    }

    printf("Quadratic x^2 + 5 * x - 14 = 0\n");
    op1.r = 1.0; op1.i = 0.0;
    op2.r = 5.0; op2.i = 0.0;
    op3.r = -14.0; op3.i = 0.0;

    if ( cplex_quadratic( quad_res, &real_root_count,
                          &op1, &op2, &op3 ) == MATH_OP_FAIL ) {

        fprintf(stderr,"WARN : cplex_quadratic returns MATH_OP_FAIL\n");

    } else {

        printf("Real root count = %i\n", real_root_count );
        printf("Quadratic result 1 = ( %16.12e, %16.12e )\n",
                                          quad_res[0].r, quad_res[0].i);
        printf("          result 2 = ( %16.12e, %16.12e )\n\n",
                                          quad_res[1].r, quad_res[1].i);

    }


    /* Wed Oct 23 2019 via IRC we have traviss who catches a bug here 
     * 
     * https://www.wolframalpha.com/input/?i=solve+for+x+where+x%5E2+-+5+*+x+%2B+14+%3D+0+
     * roots should be x = 1/2 * ( 5 +- sqrt(31)*i )
     * x = 2.50000000000000 - 2.78388218141501 i
     */
    printf("Quadratic x^2 - 5 * x + 14 = 0\n");
    op1.r = 1.0; op1.i = 0.0;
    op2.r = -5.0; op2.i = 0.0;
    op3.r = 14.0; op3.i = 0.0;

    if ( cplex_quadratic( quad_res, &real_root_count,
                          &op1, &op2, &op3 ) == MATH_OP_FAIL ) {

        fprintf(stderr,"WARN : cplex_quadratic returns MATH_OP_FAIL\n");

    } else {

        printf("Real root count = %i\n", real_root_count );
        printf("Quadratic result 1 = ( %16.12e, %16.12e )\n",
                                          quad_res[0].r, quad_res[0].i);
        printf("          result 2 = ( %16.12e, %16.12e )\n\n",
                                          quad_res[1].r, quad_res[1].i);

    }

    printf("                     ( 2.5,  2.78388218141501 )\n");
    printf("                     ( 2.5, -2.78388218141501 )\n\n");


    printf("\nQuadratic 2 * x^2 - 5 * x - 1 = 0\n");
    op1.r = 2.0;  op1.i = 0.0;
    op2.r = -5.0; op2.i = 0.0;
    op3.r = -1.0; op3.i = 0.0;
    /* styx$ echo '18k 0.25 5 33v + * pq' | dc 
     * 2.686140661634507164
     * styx$ echo '18k 0.25 5 33v - * pq' | dc 
     * -.186140661634507164
     */

    if ( cplex_quadratic( quad_res, &real_root_count,
                          &op1, &op2, &op3 ) == MATH_OP_FAIL ) {

        fprintf(stderr,"WARN : cplex_quadratic returns MATH_OP_FAIL\n");

    } else {

        printf("Real root count = %i\n", real_root_count );
        printf("Quadratic result 1 = ( %16.12e, %16.12e )\n",
                                          quad_res[0].r, quad_res[0].i);
        printf("          result 2 = ( %16.12e, %16.12e )\n\n",
                                          quad_res[1].r, quad_res[1].i);

    }
    printf("                     ( 2.686140661634507164e+00, 0.0 )\n");
    printf("                     ( -1.86140661634507164e-01, 0.0 )\n\n");



    printf("DBUG : ***********************************************\n\n");
    printf("DBUG : 201920191111020747\n");
    printf("DBUG : potapeno: has a counter example which may toss\n");
    printf("DBUG :           a monkey wrench into the workings\n");
    /* https://www.wolframalpha.com/input/?i=%281%2Bi%29x%5E2-%282%2Bi%29x%2B1%3D0 */

    printf("Quadratic ( 1 + i ) * x^2 - ( 2 + i ) * x + 1 = 0\n");
    op1.r = 1.0;  op1.i = 1.0;
    op2.r = -2.0; op2.i = -1.0;
    op3.r = 1.0; op3.i = 0.0;

    if ( cplex_quadratic( quad_res, &real_root_count,
                          &op1, &op2, &op3 ) == MATH_OP_FAIL ) {

        fprintf(stderr,"WARN : cplex_quadratic returns MATH_OP_FAIL\n");

    } else {

        printf("Real root count = %i\n", real_root_count );
        printf("Quadratic result 1 = ( %16.12e, %16.12e )\n",
                                          quad_res[0].r, quad_res[0].i);
        printf("          result 2 = ( %16.12e, %16.12e )\n\n",
                                          quad_res[1].r, quad_res[1].i);

    }
    printf("      should be ( 1, 0 ) and ( 0.5, -0.5 )\n\n");


    printf("\n-----------------------------------------------------\n");
    printf("\nDeterminant stuff follows \n");

    /* Determinant of three row matrix */
    v[0].x.r = 1.0; v[0].x.i = 0.0;
    v[0].y.r = 2.0; v[0].y.i = 0.0;
    v[0].z.r = 3.0; v[0].z.i = 0.0;
    printf("dbug : row 1 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[0].x.r, v[0].x.i,
            v[0].y.r, v[0].y.i,
            v[0].z.r, v[0].z.i );

    /* second row */
    v[1].x.r = 4.0; v[1].x.i = 0.0;
    v[1].y.r = 5.0; v[1].y.i = 0.0;
    v[1].z.r = 6.0; v[1].z.i = 0.0;
    printf("     : row 2 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[1].x.r, v[1].x.i,
            v[1].y.r, v[1].y.i,
            v[1].z.r, v[1].z.i );

    /* third row */
    v[2].x.r = 7.0; v[2].x.i = 0.0;
    v[2].y.r = 8.0; v[2].y.i = 0.0;
    v[2].z.r = 9.0; v[2].z.i = 0.0;
    printf("     : row 3 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[2].x.r, v[2].x.i,
            v[2].y.r, v[2].y.i,
            v[2].z.r, v[2].z.i );

    cplex_det(&opr, &v[0], &v[1], &v[2]); 
    printf("     : det = ( %g, %g )\n", opr.r, opr.i);
    printf("     :   should be zero\n");

    printf("\n-----------------------------------------------------\n");
    printf("New Row data on row1 \n");
    v[0].x.r = 10.0; v[0].x.i = 0.0;
    v[0].y.r = -2.0; v[0].y.i = 0.0;
    v[0].z.r = -3.0; v[0].z.i = 0.0;
    printf("dbug : row 1 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[0].x.r, v[0].x.i,
            v[0].y.r, v[0].y.i,
            v[0].z.r, v[0].z.i );
    printf("     : row 2 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[1].x.r, v[1].x.i,
            v[1].y.r, v[1].y.i,
            v[1].z.r, v[1].z.i );
    printf("     : row 3 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[2].x.r, v[2].x.i,
            v[2].y.r, v[2].y.i,
            v[2].z.r, v[2].z.i );

    cplex_det(&opr, &v[0], &v[1], &v[2]); 
    printf("     : det = ( %g, %g )\n", opr.r, opr.i);
    printf("     :   should be -33\n");

    printf("\n-----------------------------------------------------\n");
    printf("Complex row data in row1 and row 3\n");
    v[0].x.r = 0.5; v[0].x.i = -1.0;
    printf("dbug : row 1 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[0].x.r, v[0].x.i,
            v[0].y.r, v[0].y.i,
            v[0].z.r, v[0].z.i );
    v[2].y.r = -2.0; v[2].y.i = 4.0;
    printf("     : row 2 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[1].x.r, v[1].x.i,
            v[1].y.r, v[1].y.i,
            v[1].z.r, v[1].z.i );
    printf("     : row 3 = < ( %g, %g ), ( %g, %g ), ( %g, %g ) >\n",
            v[2].x.r, v[2].x.i,
            v[2].y.r, v[2].y.i,
            v[2].z.r, v[2].z.i );

    cplex_det(&opr, &v[0], &v[1], &v[2]); 
    printf("     : det = ( %g, %g )\n", opr.r, opr.i);
    printf("     :     = ( 121.5, -117 )\n");

    printf("\n-----------------------------------------------------\n");
    printf("Cramers method test where we solve for A,B,C where :\n");
    printf("  ( 0.5 - i ) * A           - 2 * B - 3 * C = ( 1 + 0.5i ),\n");
    printf("            4 * A           + 5 * B + 6 * C = ( 2 + 0.75i ),\n");
    printf("            7 * A + ( -2 + 4i ) * B + 9 * C = ( 3 - 0.25i ).\n");
    /* solutions are : 
     * alpha $ echo '18k 823 1405 / p 1533 2810 / pq' | dc
     * .585765124555160142
     * .545551601423487544
     * alpha $ echo '18k _28 1405 / p 1089 5620 / pq' | dc
     * -.019928825622775800
     * .193772241992882562
     * alpha $ echo '18k _57 1405 / p _2249 5620 / pq' | dc
     * -.040569395017793594
     * -.400177935943060498
     */

    v[0].x.r =  0.5; v[0].x.i = -1.0;
    v[0].y.r = -2.0; v[0].y.i =  0.0;
    v[0].z.r = -3.0; v[0].z.i =  0.0;

    v[1].x.r =  4.0; v[1].x.i =  0.0;
    v[1].y.r =  5.0; v[1].y.i =  0.0;
    v[1].z.r =  6.0; v[1].z.i =  0.0;

    v[2].x.r =  7.0; v[2].x.i =  0.0;
    v[2].y.r = -2.0; v[2].y.i =  4.0;
    v[2].z.r =  9.0; v[2].z.i =  0.0;

    rh_col.x.r = 1.0; rh_col.x.i = 0.5;
    rh_col.y.r = 2.0; rh_col.y.i = 0.75;
    rh_col.z.r = 3.0; rh_col.z.i = -0.25;

    if ( cplex_cramer( &res_vec, &v[0], &v[1], &v[2], &rh_col ) == MATH_OP_FAIL ) {
        printf("dbug : There is no valid solution.\n");
    } else {
        printf("     : result col = < ( %+-16.12e, %+-16.12e ),\n",
                    res_vec.x.r, res_vec.x.i );
        printf("                      ( %+-16.12e, %+-16.12e ),\n",
                    res_vec.y.r, res_vec.y.i );
        printf("                      ( %+-16.12e, %+-16.12e ) >\n\n",
                    res_vec.z.r, res_vec.z.i);
    }
    printf("     :     should be  (  5.857651245552e-01 ,  5.455516014235e-01 )\n");
    printf("     :                ( -1.992882562278e-02 ,  1.937722419928e-01 )\n");
    printf("     :                ( -4.056939501779e-02 , -4.001779359431e-01 )\n");

    printf("\n-----------------------------------------------------\n");
    printf("---------- line and plane intercept testing ---------\n\n");
    /* try a degenerate line plane intercept case where the line
     * is in the plane. */
    printf("INFO : degenerate line plane intercept with line in the plane\n");

    /* distance becomes zero */
    cplex_vec_set( &line_point, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0 );
    cplex_vec_set( &line_direction, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    cplex_vec_set( &plane_point, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0);
    cplex_vec_set( &plane_normal, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    cplex_vec_zero(&plane_u);
    cplex_vec_zero(&plane_v);

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);


    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }

    printf("\n\n--------------------------------------------------\n");

    /* try a degenerate line plane intercept case where the line
     * is in the plane. */
    printf("INFO : degenerate line plane intercept with line in the plane\n");
    printf("     : plane point and line point are identical\n");

    /* distance becomes zero */
    cplex_vec_set( &line_point, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0 );
    cplex_vec_set( &line_direction, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    cplex_vec_set( &plane_point, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0);
    cplex_vec_set( &plane_normal, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    cplex_vec_zero(&plane_u);
    cplex_vec_zero(&plane_v);

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);


    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    printf("INFO : test the line point and plane point are same\n");

    cplex_vec_set( &line_point, 0.0, 0.0, 6.0, 0.0, 3.0, 0.0);
    cplex_vec_set( &line_direction, -1.0, 0.0, 2.0, 0.0, 1.0, 0.0);
    cplex_vec_set( &plane_point, 0.0, 0.0, 6.0, 0.0, 3.0, 0.0);
    cplex_vec_set( &plane_normal, 1.0, 0.0, -3.0, 0.0, -2.0, 0.0);

    printf("     : u = NULL pointer\n");
    printf("     : v = NULL pointer\n");

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");

    /* analytic test data for the line plane intercept
     * compliments of halirutan on twitch 
     *
     *    v[0].x.r = -1.0 / sqrt(6.0);        v[0].x.i = 0.0;
     *    v[0].y.r =  0.0;                    v[0].y.i = 0.0;
     *    v[0].z.r = -1.0 * sqrt(13.0/14.0);  v[0].z.i = 0.0;
     *
     *    v[1].x.r =  2.0 / sqrt(6.0);        v[1].x.i = 0.0;
     *    v[1].y.r = -2.0 / sqrt(13.0);       v[1].y.i = 0.0;
     *    v[1].z.r = -3.0 / sqrt(182.0);      v[1].z.i = 0.0;
     *
     *    v[2].x.r =  1.0 / sqrt(6.0);        v[2].x.i = 0.0;
     *    v[2].y.r =  3.0 / sqrt(13.0);       v[2].y.i = 0.0;
     *    v[2].z.r = -1.0 * sqrt(2.0/91.0);   v[2].z.i = 0.0;
     */

    printf("\n\n--------------------------------------------------\n");
    /* create a bucket of data elements for a call to the
     * line_plane_icept() */
    cplex_vec_set( &line_point, 2.0, 0.0, 3.0, 0.0, -2.0, 0.0);
    cplex_vec_set( &line_direction, -1.0, 0.0, 2.0, 0.0, 1.0, 0.0);
    cplex_vec_set( &plane_point, 0.0, 0.0, 6.0, 0.0, 3.0, 0.0);
    cplex_vec_set( &plane_normal, 1.0, 0.0, -3.0, 0.0, -2.0, 0.0);

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    printf("     : u = NULL pointer\n");
    printf("     : v = NULL pointer\n");

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    /* try again with a zero magnitude u and v vectors */
    cplex_vec_zero(&plane_u);
    cplex_vec_zero(&plane_v);
    printf("\n\nINFO : line_plane_icept() again\n");
    printf("     : with zero mag plane_u and plane_v\n\n");

    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );
    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n\n", plane_v.z.r );

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);


    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    /* try again with a microscopic v vector where we know a 
     * valid v is 
     *   v_x = -0.96362411165943153325
     *   v_y = -0.22237479499833035382
     *   v_z = -0.14824986333222023589
     */
    cplex_vec_zero(&plane_u);
    cplex_vec_set( &plane_v, -0.96362411165943153325 / 10e10, 0.0,
                             -0.22237479499833035382 / 10e10, 0.0,
                             -0.14824986333222023589 / 10e10, 0.0 );

    printf("\n\nINFO : line_plane_icept() again\n");
    printf("     : with microscopic plane_v and zero plane_u\n");
    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );
    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n\n", plane_v.z.r );

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    /* use a microscopic v vector that is NOT in the plane */
    cplex_vec_zero(&plane_u);
    cplex_vec_set( &plane_v, -0.96362411165943153325 / 10e10, 0.0,
                              0.22237479499833035382 / 10e10, 0.0,
                              0.14824986333222023589 / 10e10, 0.0 );

    printf("\n\nINFO : line_plane_icept() again\n");
    printf("     : with microscopic plane_v NOT in the plane\n");
    printf("     : as well as a zero plane_u\n");
    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );
    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n\n", plane_v.z.r );

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    
    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    /* use a microscopic u vector that is valid as well as v vector
     * that is zero
     *
     * u_hat analytical data from Wolfram Mathematica */

    cplex_vec_zero(&plane_v);
    cplex_vec_set( &plane_u, 0.0, 0.0,
                             ( -1.0 * 2.0 / sqrt( 13.0 ) ) / 10e10, 0.0,
                             ( 3.0 / sqrt( 13.0 ) ) / 10e10, 0.0 );

    printf("\n\nINFO : line_plane_icept() again\n");
    printf("     : with microscopic plane_u and zero plane_v\n");
    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );
    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n\n", plane_v.z.r );

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    /* use a microscopic u vector that is NOT in the plane
     * also a v vector that is zero
     *
     * u_hat analytical data from Wolfram Mathematica
     * and then that was messed with slightly. */

    cplex_vec_zero(&plane_v);
    cplex_vec_set( &plane_u, 0.000000001, 0.0,
                             ( -1.0 * 2.0 / sqrt( 13.0 ) ) / 10e10, 0.0,
                             ( 3.0 / sqrt( 13.0 ) ) / 10e10, 0.0 );

    printf("\n\nINFO : line_plane_icept() again\n");
    printf("     : with microscopic and invalid plane_u\n");
    printf("     : also a zero plane_v\n");
    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );
    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n\n", plane_v.z.r );

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    /* Neither u nor v is zero in size but v vector magnitude
     * is less than RT_EPSILON. */

    cplex_vec_set( &plane_u, 0.0, 0.0,
                             ( -1.0 * 2.0 / sqrt( 13.0 ) ), 0.0,
                             ( 3.0 / sqrt( 13.0 ) ), 0.0 );

    cplex_vec_set( &plane_v, -1.0 * sqrt( 13.0 / 14.0 ) / 10e13, 0.0,
                             ( -3.0 / sqrt( 182.0 ) ) / 10e13, 0.0,
                             -1.0 * sqrt( 2.0 / 91.0 ) / 10e13, 0.0 );

    printf("\n\nINFO : line_plane_icept() again\n");
    printf("     : with perfect plane_u\n");
    printf("     : also a microscopic plane_v\n");
    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );
    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n\n", plane_v.z.r );

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    /* provide both u and v as normalized and perfectly orthogonal
     * from analytic results thanks to halirutan 
     *
     * u_hat = {0,-(2/Sqrt[13]),3/Sqrt[13]}
     *
     * v_hat = {-Sqrt[(13/14)], -(3/Sqrt[182]), -Sqrt[(2/91)]}
     *
     * nix$ echo '36k _2.0 13.0v / p  3.0 13.0v / pq' | dc
     *   -.554700196225229122018341733456999376
     *   .832050294337843683027512600185499064
     * nix$
     * nix$ echo '36k _1.0 13 14/v*p _3 182v/p _1 2 91/v*pq' | dc
     *   -.963624111659431533253593257963967497
     *   -.222374794998330353827752290299377114
     *   -.148249863332220235885168193532918076
     *   
     */
    cplex_vec_set( &plane_u, 0.0, 0.0,
                             ( -1.0 * 2.0 / sqrt( 13.0 ) ), 0.0,
                             ( 3.0 / sqrt( 13.0 ) ), 0.0 );

    cplex_vec_set( &plane_v, -1.0 * sqrt( 13.0 / 14.0 ), 0.0,
                             ( -3.0 / sqrt( 182.0 ) ), 0.0,
                             -1.0 * sqrt( 2.0 / 91.0 ), 0.0 );

    printf("\n\nINFO : line_plane_icept()\n");
    printf("     : with perfect plane_u\n");
    printf("     : also perfect plane_v\n");
    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );
    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n\n", plane_v.z.r );

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    cplex_vec_set( &plane_u, 0.0, 0.0,
                            -2.0, 0.0,
                             3.0, 0.0 );

    cplex_vec_set( &plane_v, -1.0 * sqrt( 13.0 / 14.0 ), 0.0,
                             ( -3.0 / sqrt( 182.0 ) ), 0.0,
                             -1.0 * sqrt( 2.0 / 91.0 ), 0.0 );

    printf("\n\nINFO : line_plane_icept()\n");
    printf("     : with not normalized plane_u\n");
    printf("     : also perfect plane_v\n");
    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );
    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n\n", plane_v.z.r );

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    cplex_vec_set( &plane_u, 0.0, 0.0,
                            -2.0, 0.0,
                             3.0, 0.0 );

    cplex_vec_set( &plane_v, -13.0, 0.0,
                             -3.0,  0.0,
                             -2.0,  0.0 );

    printf("\n\nINFO : line_plane_icept()\n");
    printf("     : with not normalized plane_u\n");
    printf("     : also not normalized plane_v\n");
    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );
    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n\n", plane_v.z.r );

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    cplex_vec_set( &plane_u, 0.0, 0.0,
                            -2.0, 0.0,
                             3.0, 0.0 );

    printf("\n\nINFO : line_plane_icept()\n");
    printf("     : with not normalized plane_u\n");
    printf("     : also NULL pointer plane_v\n");
    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );
    printf("     : v = NULL pointer\n\n");

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    cplex_vec_set( &plane_v, -13.0, 0.0,
                             -3.0,  0.0,
                             -2.0,  0.0 );

    printf("\n\nINFO : line_plane_icept()\n");
    printf("     : with NULL plane_u\n");
    printf("     : also not normalized plane_v\n");
    printf("     : u = NULL pointer\n");
    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n", plane_v.z.r );

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    printf("\n\nINFO : line_plane_icept()\n");
    printf("     : with NULL plane_u\n");
    printf("     : also NULL plane_v\n");
    printf("     : u = NULL pointer\n");
    printf("     : v = NULL pointer\n");

    cplex_vec_set( &line_point, -13.0, 0.0, 1.0, 0.0, 4.0, 0.0);
    printf("     : line_point = ( -13, 1, 4 )\n\n");

    printf("     :    line_point = < %g, %g, %g >\n",
            line_point.x.r, line_point.y.r, line_point.z.r);
    printf("     :      line_dir = < %g, %g, %g >\n",
            line_direction.x.r, line_direction.y.r, line_direction.z.r);
    printf("     :   plane_point = < %g, %g, %g >\n",
            plane_point.x.r, plane_point.y.r, plane_point.z.r);
    printf("     : &plane_normal = < %g, %g, %g >\n",
            plane_normal.x.r, plane_normal.y.r, plane_normal.z.r);

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    /* now a completely trivial situation */

    cplex_vec_set( &line_point, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    cplex_vec_set( &line_direction, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    cplex_vec_set( &plane_point, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0);
    cplex_vec_set( &plane_normal, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0);
    cplex_vec_set( &plane_u, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 );
    cplex_vec_set( &plane_v, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 );

    printf("\n\nINFO : line_plane_icept()\n");
    printf("     : with plane at ( 0, 8, 0)\n");
    printf("     : plane normal < 0, -1, 0>\n\n");
    printf("     : line point ( 0, 0, 0)\n");
    printf("     : line dir < 0, 1, 0>\n\n");

    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );

    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n", plane_v.z.r );

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");
    /* also trivial situation */

    cplex_vec_set( &line_point, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    cplex_vec_set( &line_direction, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    cplex_vec_set( &plane_point, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0);
    cplex_vec_set( &plane_normal, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0);
    cplex_vec_set( &plane_u, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 );
    cplex_vec_set( &plane_v, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 );

    printf("\n\nINFO : line_plane_icept()\n");
    printf("     : with plane at ( 0, 8, 0)\n");
    printf("     : plane normal < 0, -1, 0>\n\n");
    printf("     : line point ( 0, 0, 0)\n");
    printf("     : line dir < 0, 1, 0>\n\n");

    printf("     : u = %+-16.9e", plane_u.x.r);
    printf("    %+-16.9e", plane_u.y.r);
    printf("    %+-16.9e\n", plane_u.z.r );

    printf("     : v = %+-16.9e", plane_v.x.r);
    printf("    %+-16.9e", plane_v.y.r);
    printf("    %+-16.9e\n", plane_v.z.r );

    if ( line_plane_icept( &lp_intercept_point,
                           &plane_u_norm, &plane_v_norm,
                           &lp_intercept_param,
                           &line_point, &line_direction,
                           &plane_point, &plane_normal,
                           &plane_u,&plane_v) == MATH_OP_SUCCESS ) {

        printf("     : intercept = ( %-+16.9e, %-+16.9e, %-+16.9e )\n",
                                 lp_intercept_point.x.r,
                                 lp_intercept_point.y.r,
                                 lp_intercept_point.z.r);

        printf("     : u_hat = %+-16.9e", plane_u_norm.x.r);
        printf("    %+-16.9e", plane_u_norm.y.r);
        printf("    %+-16.9e\n", plane_u_norm.z.r );

        printf("     : v_hat = %+-16.9e", plane_v_norm.x.r);
        printf("    %+-16.9e", plane_v_norm.y.r);
        printf("    %+-16.9e\n\n", plane_v_norm.z.r );

    } else {
        printf("INFO : line_plane_icept() returns MATH_OP_FAIL\n");
    }


    printf("\n\n--------------------------------------------------\n");

    return ( EXIT_SUCCESS );

}

