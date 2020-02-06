
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

    int inex; /* mpfr retval */
    long delta_bit_prec, bit_prec = PREC;
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
    int input_attempt_loop = 0;

    /* input_m is whatever we are given as input and
     * then check_in and check_out will be used to compare
     * the values we get from string to mpfr_t conversion. */
    mpfr_t input_m, check_in, check_out;

    if ( setlocale( LC_ALL, "C" ) == NULL ) {
        fprintf(stderr,"FAIL : could not set LC_ALL=\"C\"\n");
        return (EXIT_FAILURE);
    }

    /* check if a test number was on the command line */
    if (argc<2) {
        fprintf(stderr,"FAIL : test number and optional precision?\n");
        fprintf(stderr,"     : %s <number> [bit precision]\n", argv[0]);
        fprintf(stderr,"     : default bit precision is %i\n\n", PREC);
        return (EXIT_FAILURE);
    }

    /* check if a bit precision parameter was on the command line */
    if ( argc > 2 ) {

        str = argv[2]; /* why do I need another char point here ? */
        bit_prec = strtol(str, &endptr, 10);

        errno = 0;
        if ( ( (errno == ERANGE) &&
               ( (bit_prec == LONG_MAX) || (bit_prec == LONG_MIN) ) 
                 ) || (errno != 0 && bit_prec == 0) ) {
            perror("FAIL : strtol could not parse bit precision.");
            exit(EXIT_FAILURE);
        }

        if (endptr == str) {
            fprintf(stderr, "No bit precision seen.\n");
            exit(EXIT_FAILURE);
        }

        if (*endptr != '\0'){ /* Not necessarily an error... */
            fprintf(stderr,"INFO : Further characters after bit");
            fprintf(stderr," precision ignored : %s\n", endptr);
        }

        if (bit_prec>1023){
            bit_prec = 1024;
            printf("WARN : start precision limited to 1024 bits.\n");
        }

    } else {
        bit_prec = PREC;
    }

    /* However we need to estimate the bit-width of the decimal
     * input value where ln(10)/ln(2) = 3.3219281 approx ratio
     * between decimal and binary precision needed. So we may assume
     * that whatever was entered in decimal will need strlen(argv[1])
     * multiplied by 3.32 and we have a guess on bit size needed. */
    input_number_len = strlen(argv[1]);
    decimal_to_bit_estimate = (long)floorf(3.32 * input_number_len);
    if ( (float)bit_prec < ( 3.32 * (float)input_number_len ) ) {
        /* at this time we have no idea if argv[1] is a decimal
         * number or some trash input. This is just a guess. */
        bit_prec = decimal_to_bit_estimate;
        printf("INFO : guess 3.3 x %i chars = %li bits needed.\n",
                input_number_len, bit_prec );
    }

    /* whatever the above bit_prec is we want it to be on a 32bit
     * boundary and thus we can just increse as needed. */
    if ( bit_prec%32 != 0 ) {
       printf("INFO : we shall adjust the requested precision to ");
       bit_prec = bit_prec + 32 - bit_prec%32;
       printf("%lu bits\n", bit_prec);
    }

    /* if we need to increase the precision then we may do so in
     * chunks that are half the current precision */
    delta_bit_prec = (bit_prec / 2);
    /* ensure delta precision is a multiple of 32bits also */
    if ( delta_bit_prec%32 != 0 ) {
        delta_bit_prec = delta_bit_prec + 32 - delta_bit_prec%32;
        printf("     : if we need to increase precision then we");
        printf(" shall do so in %lu bit increments\n", delta_bit_prec);
    }

    /* we want the decimal_to_bit_estimate to always be at least this
     * starting value bit_prec */
    if ( decimal_to_bit_estimate < bit_prec ) {
        decimal_to_bit_estimate = bit_prec;
        printf("INFO : decimal_to_bit_estimate increased to %li\n",
                decimal_to_bit_estimate );
    }
    mpfr_set_default_prec((mpfr_prec_t)bit_prec);
    mpfr_init2(input_m, (mpfr_prec_t)bit_prec);

    /* did we really get the requested precision ? */
    actual_prec=mpfr_get_default_prec();
    mpfr_printf ("     : we have precision at %Pu bits.\n",
                                                          actual_prec);

    /* setup an input buffer area for the argv[1] chars */
    /* 8-byte boundary and pad on plenty of extra room */
    if ( input_number_len%8 != 0 ) {

        input_buffer_len = input_number_len + 16
                                               - input_number_len%8;

        fprintf(stderr,"dbug : input_buffer_len set at %lu\n",
                                                     input_buffer_len);

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

    /* Why use strncat? Firstly we used calloc which assures us
     * that the memory buffer is full of zero bytes. Thus all
     * bytes in that buffer are really just terminating char nul
     * and thus it already is an empty string. We did increase
     * the size of the buffer to the next modulo 8-byte boundary
     * and thus the number of chars to copy is assured to be less
     * than the size of the buffer.
     */
    input_buf = strncat( input_buf, argv[1], input_number_len + 1);

    printf("INFO : starting input_buf = \"%s\"\n", input_buf);

    /* int mpfr_set_str(mpfr_t rop, const char *s,
     *                  int base, mpfr_rnd_t rnd)
     *
     * Set rop to the value of the string s in base base, rounded
     * in the direction rnd. See the documentation of mpfr_strtofr
     * for a detailed description of the valid string formats.
     * Contrary to mpfr_strtofr, mpfr_set_str requires the whole
     * string to represent a valid floating-point number.
     *
     * The meaning of the return value differs from other MPFR
     * functions: it is 0 if the entire string up to the final null
     * character is a valid number in base base.
     *
     * Otherwise the return is -1, and rop may have been changed
     * in this circumstance. (users interested in the ternary value
     * should use mpfr_strtofr instead).
     */


input_try:
    printf("INFO : loop %i attempt to read input", input_attempt_loop);
    if ( input_attempt_loop > 0 ) {
        printf(" again.");
    }
    printf("\n");

    fprintf(stderr,"dbug : input_buf = \"%s\"\n", input_buf);

    inex = mpfr_set_str(input_m, input_buf, 10, MPFR_RNDN);

    if ( inex < 0 ) {
        printf("FAIL : some mysterious problem happened!\n");
        printf("     : we tried to interpret %s\n", argv[1]);
        printf("     : input_buf was \"%s\"\n", input_buf);
        printf("     : however mpfr_set_str() returned \"");
        mpfr_printf("%.R*e", MPFR_RNDN, input_m);
        printf("\"\n");
        return (EXIT_FAILURE);
    }

    if (mpfr_number_p(input_m)==0) {
        fprintf(stderr,"FAIL : mpfr_set_str returned not a number.\n");
        return (EXIT_FAILURE);
    }

    if ( input_attempt_loop < 1 ) {
        printf("     : argv[1] seen as \"%s\"\n", argv[1]);
    }

    printf("     : mpfr_set_str() returns \"");
    /* mpfr_printf("%.*RNe\"\n", input_number_len, input_m); */
    mpfr_printf("%.R*e\"\n", MPFR_RNDN, input_m);

    /* int mpfr_snprintf(char *buf, size_t n, const char *template, ...
     *
     *    Form a null-terminated string corresponding to the optional
     *    arguments under the control of the template string template,
     *    and print it in buf. If n is zero, nothing is written and buf
     *    may be a null pointer, otherwise, the n-1 first characters
     *    are written in buf and the n-th is a null character. Return
     *    the number of characters that would have been written had n
     *    been sufficiently large, not counting the terminating null
     *    character, or a negative value if an error occurred. 
     */

    /* buffer plenty big enough for the string plus trailing nuls */
    output_buf = calloc( input_buffer_len + 8, sizeof(unsigned char));
    if ( output_buf == NULL ) {
        /* really? who does all this stuff? Someone wanted it in the
         * twitch stream. */
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

    /* NOTE : be very careful with the format specification and
     * do not embed a new line char or other special chars.
     *
     * From the manual : 
     *
     *     int mpfr_snprintf (char *buf, size_t n,
     *                        const char *template, ...)
     *
     *     Form a null-terminated string corresponding to the optional
     *     arguments under the control of the template string template,
     *     and print it in buf. If n is zero, nothing is written and
     *     buf may be a null pointer, otherwise, the n-1 first chars
     *     are written in buf and the n-th is a null char. Return the
     *     number of chars that would have been written had n been
     *     sufficiently large, not counting the terminating null char,
     *     or a negative value if an error occurred.
     */

    /* whatever this is ... it breaks badly and converts 0.125 to
     * the string 1.25000000etc crap 
     *
    chars_formatted = mpfr_snprintf(output_buf, input_number_len + 1,
                                                "%.*RNe",
                                                input_number_len + 1,
                                                               input_m);
                                                               */

    fprintf(stderr,"dbug : input_number_len = %lu\n",
                                                     input_number_len );
    fprintf(stderr,"dbug : input_buffer_len = %lu\n",
                                                     input_buffer_len );

    chars_formatted = mpfr_snprintf( output_buf, input_buffer_len,
                                    "%.*RNe", input_buffer_len, input_m);


    if ( chars_formatted < 0 ){
        fprintf(stderr,"FAIL : mpfr_snprintf tossed an error.\n");
        return (EXIT_FAILURE);
    }
    printf("INFO : chars_formatted = %i\n", chars_formatted);

    fprintf(stderr,"----------------------------------------------\n");
    fprintf(stderr,"dbug : output_buf = \"%s\"\n", output_buf);
    fprintf(stderr,"dbug :  input_buf = \"%s\"\n", input_buf);
    fprintf(stderr,"----------------------------------------------\n");


    /* we need to compare the two values that we have in two string
     * buffers where one of them may be scientific notation and the
     * other just an ordinary number notation.
     *
     * TODO : this seems to cause all manner of havok with numbers
     *        that are not integers.
     *
     * The only valid way to compare two floating point mpfr_t values
     * is with mpfr_cmp(mpfr_t op1, mpfr_t op2) which will return a
     * zero int value if the two operands are precisely equal.
     *
     * From the manual : 
     *
     *     Compare op1 and op2. Return a positive value if op1 > op2,
     *     zero if op1 = op2, and a negative value if op1 < op2. Both
     *     op1 and op2 are considered to their full own precision,
     *     which may differ. If one of the operands is NaN, set the
     *     erange flag and return zero.
     *
     * Note: These functions may be useful to distinguish the three
     * possible cases. If you need to distinguish two cases only, it is
     * recommended to use the predicate functions (e.g., mpfr_equal_p
     * for the equality) described below; they behave like the IEEE 754
     * comparisons, in particular when one or both arguments are NaN.
     * But only floating-point numbers can be compared (you may need to
     * do a conversion first).
     */

    if ( input_attempt_loop > 0 ) {
        /* we have to change precision and then redo the
         * comparison. Also expects the last parameter
         * to be a null pointer whose type must be mpfr_ptr */
        mpfr_clears ( check_in, check_out, (mpfr_ptr) 0 );
        /* what is the current requested precision ? */
        actual_prec=mpfr_get_default_prec();
        mpfr_printf ("     : starting precision is %Pu bits.\n",
                                                          actual_prec);
    }

    /* va list call that expects the last parameter to
     * be a null pointer (whose type must also be mpfr_ptr) */
    if ( actual_prec != decimal_to_bit_estimate ) {
        fprintf(stderr,"dbug : possible precision problem\n");
        fprintf(stderr,"     : actual_prec = ");
        mpfr_fprintf (stderr,"%Pu bits.\n", actual_prec);
        fprintf(stderr,"     : decimal_to_bit_estimate =");
        mpfr_fprintf (stderr,"%Pu bits.\n", decimal_to_bit_estimate);
    }
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
        if ( input_attempt_loop < 1 ) {
            fprintf(stderr,"     : insufficient bits of precision?\n");
        }

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

        /* gradually increase the precision */
        bit_prec = bit_prec + delta_bit_prec;
        mpfr_set_default_prec((mpfr_prec_t)bit_prec);
        /* make input variable wider */
        mpfr_set_prec(input_m,(mpfr_prec_t)bit_prec);
        actual_prec=mpfr_get_default_prec();
        printf("INFO : Input variable re-initialized with ");
        mpfr_printf ("%Pu bits of precision.\n", actual_prec);
        /* clear the output buffer and start over */
        free(output_buf);
        output_buf = NULL;
        input_attempt_loop += 1;
        if ( input_attempt_loop > 4 ) {
            fprintf(stderr,"FAIL : too many loops.\n");
            return ( EXIT_FAILURE );
        }
        goto input_try;
    }

    free(output_buf);
    output_buf = NULL;

    free(input_buf);
    input_buf = NULL;

    mpfr_clears ( input_m, check_in, check_out, (mpfr_ptr) 0 );
    return (EXIT_SUCCESS);

}

