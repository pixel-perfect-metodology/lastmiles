
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
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <locale.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <errno.h>
#include <math.h>

#include <fenv.h>
#pragma STDC FENV_ACCESS ON

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

/* On a live stream we attempt to implement a tool 
 * that will convert an input fp value to a binary
 * 32 bit IEEE-754 2008 format.  Which has a sign
 * bit and 8-bits for an offset exponent and then
 * a remainder of 22 significand bits wherein there
 * is an implied leading 1 bit. */

uint64_t system_memory(void);
int sysinfo(void);

int main ( int argc, char *argv[] )
{

    double candidate_double, num;
    double exponent_calc, running_total, test_subtract;
    int j, actual_exponent, fp32_exponent, sign_bit = 0;
    int fpe_raised = 0;
    uint8_t bit_mask, shifter;

    /* some sort of a place to shove the bits */
    uint32_t fp32_bit_mask, fp32_register = 0;
    uint32_t one = 1;

    if ( argc < 2 ) {
        fprintf(stderr,"FAIL : provide a decimal number\n");
        return ( EXIT_FAILURE );
    }

    sysinfo();

    errno = 0;
    feclearexcept(FE_ALL_EXCEPT);
    candidate_double = strtod(argv[1], (char **)NULL);
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if (fpe_raised!=0){
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
    }
    if ( fpe_raised & FE_INEXACT ) {
        printf("real : Perfectly safe to ignore FE_INEXACT\n");
    }
    if ( ( errno == ERANGE ) || ( errno == EINVAL ) ){
        fprintf(stderr,"FAIL : number not understood\n");
        perror("     ");
        return ( EXIT_FAILURE );
    }
    if ( !isnormal(candidate_double) && ( candidate_double != 0.0 ) ) {
        fprintf(stderr,"FAIL : number is not normal\n");
        fprintf(stderr,"     : looks like %-+22.16e\n", candidate_double);
        return ( EXIT_FAILURE );
    }
    feclearexcept(FE_ALL_EXCEPT);
    num = candidate_double;

    /* num = strtod( argv[1], (char **)NULL ); */
    printf ("INFO : seems like a decimal number %-+22.16g\n", num);

    if ( num < 0.0 ){
        sign_bit = 1;
        fp32_bit_mask = one<<31;
        /* we only have the sign of this thing */
        fp32_register = fp32_bit_mask;
        num = fabs(num);
        printf ("     : a negative number will have sign_bit = 1\n");
    }

    /*
     * An example would be pi = 3.14159265358979323846264..
     *                          3.14159265358979323846264338327950...
     * 
     * The IEEE-754 32-bit value would be 0x40490fdb
     *
     *       0100 0000 0100 1001 0000 1111 1101 1011
     *       ^ sign bit is zero for a positive number
     *
     *
     *
     *        100 0000 0100 1001 0000 1111 1101 1011
     *        ^^^ ^^^^ ^
     *        These are the offset 127 exponent bits
     *        and here we see 10000000 == 128 decimal
     *        so we subtract the 127 offset to get 1.
     *        Therefore the exponent is really just 1.
     *
     *                  100 1001 0000 1111 1101 1011
     *                  All that remains in the actual
     *                  floating point significand.
     *
     * Note that there is always an implied value of 1
     * in front of the significand on normal numbers.
     *
     * Thus the significand is really : 
     *
     *                1.100 1001 0000 1111 1101 1011
     *
     *
     */

    exponent_calc = floor( log(num)/log(2.0) );
    actual_exponent = (int) exponent_calc;
    printf ("     : actual exponent should be %i\n", actual_exponent );

    fp32_exponent = actual_exponent + 127;
    printf ("     : binary fp32 exponent will be ");
    for ( j=7; j>=0; j-- ){
        bit_mask = (uint8_t)(one<<j);
        shifter = ( fp32_exponent & bit_mask ) ? 1 : 0;
        if (shifter){
            printf("1");
            fp32_bit_mask = one<<(j+23);
            fp32_register = fp32_register | fp32_bit_mask;
        }else{
            printf("0");
        }
    }
    printf("\n");

    /* Manually using dc to attempt to compute the bits in
     * the significand : 
     *
     * $ echo '36k 36.584 32 /  1 - p 2 _3^ - p 2 _6^ - p
     *                                     2 _9^ - p 2 _11^ - p
     *                                     2 _13^ - p  2_14^-p
     *                                     2_21^-p 2_27^-p 
     *                                     q' | dc 
     * .143250000000000000000000000000000000
     * .018250000000000000000000000000000000
     * .002625000000000000000000000000000000
     * .000671875000000000000000000000000000
     * .000183593750000000000000000000000000
     * .000061523437500000000000000000000000
     * .000000488281250000000000000000000000
     * .000000011444091796875000000000000000
     * .000000003993511199951171875000000000
     * $  
     *
     * Reference data for 36.584 in various formats : 
     *
     *     32-bit float       0x42125604
     *
     *     64-bit double      0x40424ac083126e98
     *
     *    128-bit long double 0x400424ac083126e978d4fdf3b645a1cb
     *
     */

    running_total = num / pow( 2.0, exponent_calc) - 1.0;
    printf("INFO : starting point running_total = %-+22.16e\n", running_total );
    for ( j=0; j<23; j++ ){
        test_subtract = pow( 2.0, ( 0.0 - (double)( j + 1 ) ) );
        printf("INFO : test bit %02i with value %-+22.16e\n", j+1, test_subtract);
        if ( ( running_total - test_subtract ) > 0.0 ) {
            running_total = running_total - test_subtract;
            printf("     : good bit %02i with new   %-+22.16e\n", j+1, running_total );
            fp32_register = fp32_register | ( one<<(22-j) );
        }
    }

    printf("INFO : perhaps fp32_register = 0x%8X\n", fp32_register );

    return ( EXIT_SUCCESS );

}

