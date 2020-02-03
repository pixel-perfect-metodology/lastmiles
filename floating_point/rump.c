
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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fenv.h>
#pragma STDC FENV_ACCESS ON
#include <float.h>


/* this looks like a cool attempt to squeeze some 
 * precision out of the dark matter of the universe
 * but really ya just can not do that. Sorry. */
long double shift_in_fp( long double a, long double b)
{
        long double big_number = powl(b, 6.0L) + powl(a, 2.0L)
                                   * ( 11.0L * powl(a, 2.0L)
                                       * powl(b, 2.0L) 
                                       - powl(b, 6.0L)
                                       - 121.0L * powl(b, 4.0L) - 2.0L )
                                       + 5.5L * powl(b, 8.0L) + (a / (2.0L * b ) );

        long double shift_value = powl(2.0L, logl(fabsl(big_number)) / logl(2.0) );

        return ( 333.75L * shift_value * big_number / shift_value );

}

int main(int argc, char **argv)
{

    /* see page 13 of The Handbook of Floating Point Arithmetic
     *
     * where we see that the actual result should be 
     *
     * vesta$ bc -l
     * scale=48
     * a = 77617
     * b = 33096
     * f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *         + 5.5 * b^8 + ( a / ( 2 * b ) )
     * 
     * -0.827396059946821368141165095479816291999033115785
     * 
     *  we will most likely see some other result from IEEE-754 2008 
     *  type floating point math. 
     *
     */


    /* the schmide test */
    printf( "\n     : schmide  test = %-+32.28e\n\n",
                                     shift_in_fp( 77617.0, 33096.0 ) );

    long double a, b, f, tmp[12];
    int fp_status, fp_round_mode, fpe_raised;

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

    fp_status = fesetround( FE_TONEAREST );
    assert( fp_status == 0 );

    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");

    a = 77617.0L;
    b = 33096.0L;

    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *      + 5.5 * b^8 + ( a / ( 2 * b ) ) */

    f= 333.75 * powl(b,6.0)
        + powl(a,2.0)
            * ( 11.0 * powl(a,2.0) * powl(b,2.0) 
                 - powl(b,6.0) - 121.0 * powl(b,4.0) - 2.0 )
        + 5.5 * powl(b,8.0) + ( a / ( 2.0 * b ) );

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

    printf( "\n     : f = %-+32.28e\n\n", f );
    printf( "INFO : f = -0.827396059946821368141165095 ?\n\n");

    printf( "\n----------- start over slowly -----------\n\n");


    printf("     : clear all floating point exception flags\n");
    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");


    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *      + 5.5 * b^8 + ( a / ( 2 * b ) ) */

    tmp[0] = 333.75L * b * b * b * b * b * b;
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[0] = 333.75L * b * b * b * b * b * b\n");
    printf("     : tmp[0] = %-+42.36Le\n", tmp[0]);

    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");

    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *    ^..tmp[0]..^
     *
     *
     *      + 5.5 * b^8 + ( a / ( 2 * b ) )
     *
     */

    tmp[2] = 11.0L * a * a;
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[2] = 11.0 * a * a\n");
    printf("     : tmp[2] = %-+42.36Le\n", tmp[2]);

    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");


    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *    ^.. tmp[0].^           ^.tmp[2]
     *
     *      + 5.5 * b^8 + ( a / ( 2 * b ) ) */

    tmp[3] = tmp[2] * b * b;
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[3] = tmp[2] * b * b\n");
    printf("     : tmp[3] = %-+42.36Le\n", tmp[3]);

    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");


    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *    ^ tmp[0] ^             ^.. tmp[3]...^
     *
     *
     *      + 5.5 * b^8 + ( a / ( 2 * b ) )
     *
     */

    tmp[4] = b * b * b * b * b * b;
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[4] = b * b * b * b * b * b\n");
    printf("     : tmp[4] = %-+42.36Le\n", tmp[4]);

    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");


    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *    ^ tmp[0] ^             ^.. tmp[3] ..^   tmp[4]
     *
     *
     *      + 5.5 * b^8 + ( a / ( 2 * b ) ) */

    tmp[5] = 121.0L * b * b * b * b - 2.0L;
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[5] = 121.0 * b * b * b * b - 2.0\n");
    printf("     : tmp[5] = %-+42.36Le\n", tmp[5]);

    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");

    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *    ^ tmp[0] ^             ^.. tmp[3] ..^   tmp[4]  ^.. tmp[5]^
     *
     *
     *      + 5.5 * b^8 + ( a / ( 2 * b ) ) */

    tmp[1] = tmp[3] - tmp[4] - tmp[5];
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[1] = tmp[3] - tmp[4] - tmp[5]\n");
    printf("     : tmp[1] = %-+42.36Le\n", tmp[1]);

    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");

    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *    ^ tmp[0] ^             ^............. tmp[1] .............^
     *
     *
     *      + 5.5 * b^8 + ( a / ( 2 * b ) ) */

    tmp[6] = tmp[1] * a * a;
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[6] = tmp[1] * a * a\n");
    printf("     : tmp[6] = %-+42.36Le\n", tmp[6]);
    printf("     : correct= -7917111779274712207494296608131179134\n");

    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");

    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *    ^ tmp[0] ^     ^............... tmp[6] ...................^
     *
     *
     *      + 5.5 * b^8 + ( a / ( 2 * b ) ) */


    tmp[7] = 5.5L * b * b * b * b * b * b * b * b;
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[7] = 5.5L * b * b * b * b * b * b * b * b\n");
    printf("     : tmp[7] = %-+42.36Le\n", tmp[7]);

    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *    ^ tmp[0] ^     ^............... tmp[6] ...................^
     *
     *
     *      + 5.5 * b^8 + ( a / ( 2 * b ) )
     *        ^ tmp[7]^
     *
     *
     */

    /* lets check if powl can do b^8 perfectly *
    tmp[7] = powl( b, 8.0 );
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[7] = powl( b, 8.0 )\n");


    printf("INFO : how borked is powl( b, 8.0 )?\n");
    printf("     : tmp[6] = %-+42.36Le\n", tmp[6]);
    printf("     : tmp[7] = %-+42.36Le\n", tmp[7]);
    if ( ( tmp[6] - tmp[7] ) == 0.0L )
        printf("DBUG : well gee tmp[6] equals tmp[7]\n");
    */

    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");

    tmp[8] = a / ( 2.0L * b );
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[8] = a / ( 2.0L * b )\n");
    printf("     : tmp[8] = %-+42.36Le\n", tmp[8]);

    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *    ^ tmp[0] ^     ^............... tmp[6] ...................^
     *
     *
     *      + 5.5 * b^8 + ( a / ( 2 * b ) )
     *        ^ tmp[7]^    ^... tmp[8]...^
     *
     */

    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");

    tmp[9] = tmp[7] + tmp[8];
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[9] = tmp[7] + tmp[8]\n");
    printf("     : tmp[9] = %-+42.36Le\n", tmp[9]);

    /* f= 333.75 * b^6 + a^2 * ( 11 * a^2 * b^2 - b^6 - 121 * b^4 - 2 )
     *    ^ tmp[0] ^     ^............... tmp[6] ...................^
     *
     *
     *      + 5.5 * b^8 + ( a / ( 2 * b ) )
     *        ^ ....... tmp[9] ..........^
     *
     */


    if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
        printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
    } else {
        printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
        return ( EXIT_FAILURE );
    }
    printf("\n");

    tmp[10] = tmp[0] + tmp[6] + tmp[9];
    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if ( fpe_raised != 0 ) {
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
        if ( feclearexcept(FE_ALL_EXCEPT) == 0 ) {
            printf("     : feclearexcept(FE_ALL_EXCEPT) done\n");
        } else {
            printf("\nFAIL : feclearexcept(FE_ALL_EXCEPT) fails\n");
            return ( EXIT_FAILURE );
        }
    } else {
        printf("     : precise result\n");
    }
    printf("     : done tmp[10] = tmp[0] + tmp[6] + tmp[9]\n");
    printf("     : tmp[10] = %-+42.36Le\n", tmp[10]);


    return ( EXIT_SUCCESS );

}

