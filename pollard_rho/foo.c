
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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <string.h>
#include <strings.h>
#include <locale.h>
#include <math.h>
#include <gmp.h>
#include <mpfr.h>

/* Our default starting point precision will be 64 bits
 * however this is not at all the same as IEEE 754-2008
 * double type which is 64 bits in size but uses 53 bits
 * for the effective significand. Actually 52 bits with an
 * implied leading 1 for normal floating point data. This
 * MPFR datatype will use all 64-bits for the significand. */
#define PREC 64

int main (int argc, char *argv[])
{

    int inex; /* general mpfr retval */
    long bit_prec = PREC;
    long decimal_to_bit_estimate;
    mpfr_prec_t actual_prec; /* see _MPFR_PREC_FORMAT */
    char *endptr, *str;

    /* buffer to compare what we detect at a given bit precision */
    char *output_buf;

    /* also we may need to copy input argv[1] to a place that can
     * be modified */
    char *input_buf;
    size_t input_number_len, input_buffer_len;
    int chars_formatted;

    /* input_m is whatever we are given as input and
     * then check_in and check_out will be used to compare
     * the values we get from string to mpfr_t conversion. */
    mpfr_t input_m, check_in, check_out;

    if ( setlocale( LC_ALL, "C" ) == NULL ) {
        fprintf(stderr,"FAIL : could not set LC_ALL=\"C\"\n");
        return (EXIT_FAILURE);
    }

    printf("GMP  library version : %d.%d.%d\n",
            __GNU_MP_VERSION,
            __GNU_MP_VERSION_MINOR,
            __GNU_MP_VERSION_PATCHLEVEL );

    printf("MPFR library: %-12s\n", mpfr_get_version ());
    printf("MPFR header : %s (based on %d.%d.%d)\n",
            MPFR_VERSION_STRING,
            MPFR_VERSION_MAJOR,
            MPFR_VERSION_MINOR,
            MPFR_VERSION_PATCHLEVEL);

    if (mpfr_buildopt_tls_p()!=0)
        printf("            : compiled as thread safe using TLS\n");

    if (mpfr_buildopt_float128_p()!=0)
        printf("            : __float128 support enabled\n");

    if (mpfr_buildopt_decimal_p()!=0)
        printf("            : decimal float support enabled\n");

    if (mpfr_buildopt_gmpinternals_p()!=0)
        printf("            : compiled with GMP internals\n");

    if (mpfr_buildopt_sharedcache_p()!=0)
        printf("            : threads share cache per MPFR const\n");

    printf("MPFR thresholds file used at compile time : %s\n",
                                      mpfr_buildopt_tune_case ());

    /* check if a test number was on the command line */
    if (argc<2) {
        fprintf(stderr,"FAIL : test number needed\n");
        fprintf(stderr,"     : %s <number>\n", argv[0]);
        fprintf(stderr,"     : default bit precision is %i\n\n", PREC);
        return (EXIT_FAILURE);
    }

    input_number_len = strlen(argv[1]);
    /* at this time we have no idea if argv[1] is a decimal
     * number or some trash input. Lets hope the user is not
     * a moron. */
    decimal_to_bit_estimate = (long)floorf(3.32 * input_number_len);
    if ( bit_prec < decimal_to_bit_estimate ) {
        bit_prec = decimal_to_bit_estimate;
        printf("INFO : guess 3.3 bits per decimal digit\n");
        printf("     : precision increased to %li bits\n", bit_prec );
    }

    /* whatever the above bit_prec is we want it to be on a 32bit
     * boundary and thus we can just increse as needed. */
    if ( bit_prec%32 != 0 ) {
       bit_prec = bit_prec + 32 - bit_prec%32;
    }
    printf("     : precision increased to %li bits\n", bit_prec );

    mpfr_set_default_prec((mpfr_prec_t)bit_prec);
    mpfr_init2(input_m, (mpfr_prec_t)bit_prec);

    /* did we really get the requested precision ? */
    actual_prec=mpfr_get_default_prec();
    if ( actual_prec != (mpfr_prec_t)bit_prec ) {
        fprintf(stderr,"FAIL : requested bit precision.\n");
        mpfr_fprintf (stderr,"     : request %Pu bits fails.\n",
                                                          actual_prec);
        return ( EXIT_FAILURE );
    }

    /* setup an input buffer area for the argv[1] chars */
    /* 8-byte boundary and pad on plenty of extra room */
    if ( input_number_len%8 != 0 ) {

        input_buffer_len = input_number_len + 8 - input_number_len%8;

    } else {
        /* pad on another 8 bytes regardless */
        input_buffer_len = input_number_len + 8;
    }

    printf("INFO : input number should fit well into %li bytes.\n",
                                                     input_buffer_len);

    errno = 0;
    input_buf = calloc( input_buffer_len, sizeof(unsigned char));
    if ( input_buf == NULL ) {
        /* really? possible ENOMEM? */
        if ( errno == ENOMEM ) {
            fprintf(stderr,"FAIL : calloc returns ENOMEM at %s:%d\n",
                    __FILE__, __LINE__ );
        } else {
            fprintf(stderr,"FAIL : calloc fails at %s:%d\n",
                    __FILE__, __LINE__ );
        }
        perror("FAIL ");
        return ( EXIT_FAILURE );
    }

    /* n.b. : we use calloc to get nul filled buffers */
    input_buf = strncat( input_buf, argv[1], input_number_len);

    printf("INFO : starting input_buf = \"%s\"\n", input_buf);

    if ( mpfr_set_str(input_m, input_buf, 10, MPFR_RNDN) < 0 ) {
        printf("FAIL : some mysterious problem happened!\n");
        printf("     : we tried to interpret %s\n", argv[1]);
        printf("     : input_buf was \"%s\"\n", input_buf);
        printf("     : however mpfr_set_str() returned \"");
        mpfr_printf("%.R*e", MPFR_RNDN, input_m);
        printf("\"\n");
        return (EXIT_FAILURE);
    }

    if ( mpfr_number_p(input_m) == 0 ) {
        fprintf(stderr,"FAIL : mpfr_set_str returned not a number.\n");
        return (EXIT_FAILURE);
    }

    printf("TEST : mpfr_set_str() returns \"");
    mpfr_printf("%.R*e\"\n", MPFR_RNDN, input_m);

    /* buffer plenty big enough for the input number expressed in
     * scientific notation plus trailing nuls */
    output_buf = calloc( input_buffer_len + 16, sizeof(unsigned char));
    if ( output_buf == NULL ) {
        /* really? */
        if ( errno == ENOMEM ) {
            fprintf(stderr,"FAIL : calloc returns ENOMEM at %s:%d\n",
                    __FILE__, __LINE__ );
        } else {
            fprintf(stderr,"FAIL : calloc fails at %s:%d\n",
                    __FILE__, __LINE__ );
        }
        perror("FAIL ");
        return ( EXIT_FAILURE );
    }

    fprintf(stderr,"dbug : input_number_len = %lu\n", input_number_len );
    fprintf(stderr,"dbug : input_buffer_len = %lu\n", input_buffer_len );

    /* int mpfr_snprintf (char *buf, size_t n, const char *template, ...);
     *
     * Form a null-terminated string corresponding to the optional
     * arguments under the control of the template string template, and
     * print it in buf. If n is zero, nothing is written and buf may be
     * a null pointer, otherwise, the n-1 first characters are written
     * in buf and the n-th is a null character. Return the number of
     * characters that would have been written had n been sufficiently
     * large, not counting the terminating null character, or a negative
     * value if an error occurred. 
     */
    chars_formatted = mpfr_snprintf( output_buf, input_buffer_len + 12,
                                                 "%.*RNe",
                                                 input_buffer_len,
                                                 input_m);

    if ( chars_formatted < 0 ){
        fprintf(stderr,"FAIL : mpfr_snprintf tossed an error.\n");
        return (EXIT_FAILURE);
    }
    printf("INFO : chars_formatted = %i\n", chars_formatted);

    fprintf(stderr,"----------------------------------------------\n");
    fprintf(stderr,"dbug : output_buf = \"%s\"\n", output_buf);
    fprintf(stderr,"dbug :  input_buf = \"%s\"\n", input_buf);
    fprintf(stderr,"----------------------------------------------\n");

    mpfr_inits2 ( actual_prec, check_out, (mpfr_ptr) 0 );
    mpfr_inits2 ( decimal_to_bit_estimate, check_in, (mpfr_ptr) 0 );

    inex = mpfr_set_str( check_in, input_buf, 10, MPFR_RNDN);
    if ( inex < 0 ) {
        printf("FAIL : some mysterious problem happened!\n");
        printf("     : we tried to interpret %s\n", input_buf);
        printf("     : however mpfr_set_str() returned \"");
        mpfr_printf("%.R*e", MPFR_RNDN, check_in);
        printf("\"\n");
        return (EXIT_FAILURE);
    }

    inex = mpfr_set_str( check_out, output_buf, 10, MPFR_RNDN);
    if ( inex < 0 ) {
        printf("FAIL : some mysterious problem happened!\n");
        printf("     : we tried to interpret %s\n", output_buf);
        printf("     : however mpfr_set_str() returned \"");
        mpfr_printf("%.R*e", MPFR_RNDN, check_out);
        printf("\"\n");
        return (EXIT_FAILURE);
    }

    inex = mpfr_cmp( check_in, check_out );
    if ( inex == 0 ) {
        printf("INFO : perfect match?\n");
        printf("\n\n-------------------------------------------\n\n");
        printf("INFO : here is the actual value of check_out\n\n");
        /* lets dump out the entire contents */
        mpfr_printf("%.R*e", MPFR_RNDN, check_out);
        printf("\n\n-------------------------------------------\n\n");
    } else {
        fprintf(stderr,"WARN : mpfr_cmp() comparison not equal\n");

        fprintf(stderr,"dbug : mpfr_cmp returns %+i\n",inex<0? -1 : 1);
        if ( inex < 0 ) {
            fprintf(stderr,"     : check_in < check_out\n");
        } else {
            fprintf(stderr,"     : check_out < check_in\n");
        }
        fprintf(stderr,"     : where check_in seems to be ");
        mpfr_fprintf(stderr,"%.R*e\n", MPFR_RNDN, check_in);
        fprintf(stderr,"     : check_out claims to be ");
        mpfr_fprintf(stderr,"%.R*e\n", MPFR_RNDN, check_out);

        /* TODO we may be caught in a situation where the scientific
         * notation output does not match the input string. */
    }

    free(output_buf);
    output_buf = NULL;

    free(input_buf);
    input_buf = NULL;

    mpfr_clears ( input_m, check_in, check_out, (mpfr_ptr) 0 );
    return (EXIT_SUCCESS);

}

