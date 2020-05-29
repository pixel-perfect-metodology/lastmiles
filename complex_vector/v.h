
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

#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

#define PI_L  3.141592653589793238462643383279502884L

/* for cube roots and deMoivre's Theorem we need 2pi / 3 */
#define PI3_L 2.0943951023931954923084289221863352533333L

/* Given that this is a Ray Trace prototype we may limit
 * ourselves to an epsilon for zero testing wherein anything
 * smaller than 10^(-12) is essentially zero. */
#define RT_EPSILON 1.0e-12

/* It may be senseless to consider angles that are near
 * a millionth of a degree or about 2.980232238770e-08 rad
 * and we may as well limit at twice that angle. */
#define RT_ANGLE_EPSILON 5.96046447754e-08
/* There may be a situation where we test for nearly
 * orthogonal angles in which case the deviation from
 * perfect cos(theta) = 1 may be a truely small amount
 * that coincides with RT_ANGLE_EPSILON variation */
#define RT_ANGLE_COS_EPSILON 1.776356839400809e-15

typedef struct cplex {
    double r, i;
} cplex_type;

typedef struct vec {
    struct cplex x, y, z;
} vec_type;

/* The lowest level data inside a pixel will have red, green, blue
 * and alpha data as well as luminosity. For now we will just go
 * with a luminosity between zero and one packed into a 32-bit int */
typedef struct pix_e {
    uint32_t lum;    /* 32bit luminosity */
} pix_e_type;

/* a pixel is actually a set of samples in a 3x3 grid */
typedef struct pixel {
    pix_e_type p[3][3];
} pixel_type;

int cplex_add( cplex_type *res, cplex_type *op1, cplex_type *op2 );
int cplex_sub( cplex_type *res, cplex_type *op1, cplex_type *op2 );
int cplex_mult( cplex_type *res, cplex_type *op1, cplex_type *op2 );
int cplex_div( cplex_type *res, cplex_type *op1, cplex_type *op2 );
int cplex_sq( cplex_type *res, cplex_type *op1 );
int cplex_sqrt( cplex_type *res, cplex_type *op1 );
int cplex_cbrt( cplex_type *res, cplex_type *op1 );

int cplex_vec_set( vec_type *op, double r0, double i0,
                                 double r1, double i1,
                                 double r2, double i2 );

int cplex_vec_print( vec_type *op );
int cplex_vec_zero( vec_type *op );
int cplex_vec_copy( vec_type *dst, vec_type *src);
int cplex_vec_add( vec_type *res, vec_type *op1, vec_type *op2 );
int cplex_vec_scale( vec_type *res, vec_type *op1, double factor );
int cplex_vec_dot( cplex_type *res, vec_type *op1, vec_type *op2 );
int cplex_vec_cross( vec_type *res, vec_type *op1, vec_type *op2 );
int cplex_vec_normalize( vec_type *res, vec_type *op1 );

int cplex_quadratic( cplex_type res[4],
                     cplex_type *op1,
                     cplex_type *op2,
                     cplex_type *op3 );

int cplex_det( cplex_type *res,
               vec_type *r1,
               vec_type *r2,
               vec_type *r3 );


int cplex_cramer( vec_type *res,
                  vec_type *r1,
                  vec_type *r2,
                  vec_type *r3,
                  vec_type *d );

double cplex_mag( cplex_type *op1 );
double cplex_theta( cplex_type *op1 );
double cplex_vec_mag( vec_type *op1 );

/* return the solutions to the complex coefficient quadratic */
int icept( cplex_type res[2],
           vec_type *sign,
           vec_type *loc,
           vec_type *axi,
           vec_type *obs_p,
           vec_type *obs_v );

/* compute an actual intercept point if possible */
int surface_icept_pt( vec_type *pt,
                      int intercept_cnt,
                      cplex_type *k_val,
                      vec_type *obs_point,
                      vec_type *ray_direction);

/* compute intercept for a line and plane */
int line_plane_icept( vec_type *icept_pt,
                      vec_type *plun,
                      vec_type *plvn,
                      vec_type *kst,
                      vec_type *lp0, vec_type *lpr,
                      vec_type *pl0, vec_type *pn,
                      vec_type *plu, vec_type *plv);

/* compute the surface gradient normal vector */
int gradient( vec_type *res,
              vec_type *sign,
              vec_type *loc,
              vec_type *axi,
              vec_type *intercept );

int cplex_check( cplex_type *op );
int check_dot(cplex_type *dat);
size_t tohex( char **ret, const void *addr, const size_t n );
int double_cmp( double *a, double *b, size_t n);
int vec_cmp( vec_type *a, vec_type *b, size_t n);
