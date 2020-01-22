
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
#include <math.h>
#include <fenv.h>
#pragma STDC FENV_ACCESS ON
#include <float.h>

int main()
{

    /* see page 13 of The Handbook of floating Point Arithmetic
     *
     * where we see that the actual result should be 
     *
     * vesta$ bc -l
     * scale=48
     * a = 77617
     * b = 33096
     * f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 ) + 5.5 * b^8 + ( a / ( 2 * b ) )
     * f
     * -.827396059946821368141165095479816291999033115785
     * 
     *  we will most likely see some other result from IEEE-754 2008 
     *  type floating point math. 
     *
     */

    double a, b, f;
    int fp_round_mode, fpe_raised;

#ifdef FLT_EVAL_METHOD
    printf ( "INFO : FLT_EVAL_METHOD == %d\n", FLT_EVAL_METHOD);
#endif

    fp_round_mode = fegetround();
    /* printf("DBUG : fp_round_mode = 0x%08x\n", fp_round_mode ); */
    printf("     : fp rounding mode is ");
    switch(fp_round_mode){
        case FE_TONEAREST:
            printf("FE_TONEAREST\n");
            break;
        case FE_TOWARDZERO:
            printf("FE_TOWARDZERO\n");
            break;
        case FE_UPWARD:
            printf("FE_UPWARD\n");
            break;
        case FE_DOWNWARD:
            printf("FE_DOWNWARD\n");
            break;
        default:
            printf("unknown!\n");
            break;
        }

    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");

    a = 77617.0;
    b = 33096.0;
    f= 333.75 * pow(b,6.0)
        + pow(a,2.0)
            * ( 11.0 * pow(a,2.0) * pow(b,2.0) 
                 - pow(b,6.0) - 121.0 * pow(b,4.0) - 2.0 )
        + 5.5 * pow(b,8.0) + ( a / ( 2.0 * b ) );

    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if (fpe_raised!=0){
        printf("INFO : FP Exception raised is");
        if(fpe_raised & FE_INEXACT) printf(" FE_INEXACT");
        if(fpe_raised & FE_DIVBYZERO) printf(" FE_DIVBYZERO");
        if(fpe_raised & FE_UNDERFLOW) printf(" FE_UNDERFLOW");
        if(fpe_raised & FE_OVERFLOW) printf(" FE_OVERFLOW");
        if(fpe_raised & FE_INVALID) printf(" FE_INVALID");
        printf("\n");
    } else {
        printf(" nothing!\n");
    }

    printf( "\n     : f = %-+32.28e\n\n", f );

    printf( "INFO : f = -0.827396059946821368141165095 ?\n\n");


    return ( EXIT_SUCCESS );

}

