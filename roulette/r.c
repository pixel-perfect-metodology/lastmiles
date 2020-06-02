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
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <errno.h>

/* Note : the american style double zero wheel has numbers 
 *        arranged thus : 
 *         0 28  9 26 30 11  7 20 32 17  5 22 34 15  3 24 36 13  1
 *        00 27 10 25 29 12  8 19 31 18  6 21 33 16  4 23 35 14  2
 *
 * The layout generally looks like this : 

    ---------------------------------------------------------------------------+
   /   |     |     |     |     |     |     |     |     |     |     |     |     |
  /    |  3r |  6b |  9r | 12r | 15b | 18r | 21r | 24b | 27r | 30r | 33b | 36r |
 /     |     |     |     |     |     |     |     |     |     |     |     |     |
+  00  +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
|      |     |     |     |     |     |     |     |     |     |     |     |     |
+------+  2b |  5r |  8b | 11b | 14r | 17b | 20b | 23r | 26b | 29b | 32r | 35b |
|      |     |     |     |     |     |     |     |     |     |     |     |     |
+   0  +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 \     |     |     |     |     |     |     |     |     |     |     |     |     |
  \    |  1r |  4b |  7r | 10b | 13b | 16r | 19r | 22b | 25r | 28b | 31b | 34r |
   \   |     |     |     |     |     |     |     |     |     |     |     |     |
    ---------------------------------------------------------------------------+

     */

int main ( int argc, char *argv[] ) { 

    /* The ball[] array is a record of the number of times that the 
     * roulette wheel ball lands on a given number with ball[0] being
     * the single 0 and ball[1] being the double 00.  Therefore we
     * have thirty eight possible places the ball can land.
     */
    uint32_t ball[38];

    /* This next array lets us know what colour the ball 
     * had landed on top of.  A zero indicates black and a one
     * indicates red colour.  The zero and double zero are 
     * usually green in colour and we don't need to worry about
     * them unless we land on them of course.  This array starts
     * at the number one and goes up to thirty six.
     *
     * The table seems to be laid out in a simple X pattern made
     * of red coloured squares within a three by three box of
     * numbers and this pattern is repeated four times. That would
     * result in too many red squares because each three by three
     * pattern has five reds and four blacks.  Thus the numbers
     * ten and twenty eight are both made black to even out the
     * distribution.
     */
    static uint8_t colour_data[36] = { 
            1,       0,       1, 
            0,       1,       0,
            1,       0,       1,

            0,       0,       1, 
            0,       1,       0,
            1,       0,       1,

            1,       0,       1, 
            0,       1,       0,
            1,       0,       1,

            0,       0,       1, 
            0,       1,       0,
            1,       0,       1
    };

    uint32_t bankroll, bankroll_start, profit_limit, bet, max_spin;
    uint32_t n_even, n_odd, n_red, n_black, n_zero, colour_flag;
    uint32_t iteration_count, expected_n, i, j, k;
    FILE   * fp;
    double rval, rho;

    if ( argc < 5 ) {
        printf ( "-------------------------------------" );
        printf ( "------------------------------------\n" );
        printf ( "usage : %s bankroll bet profit_limit", argv[0] );
        printf ( " spin_limit\n" );
        printf ( "      : Where the \"bankroll\" is the" );
        printf ( " amount you have to begin with\n" );
        printf ( "      : and \"bet\" is the amount of each bet" );
        printf ( " made. The \"profit_limit\"\n" );
        printf ( "      : is the amount of winnings at which you" );
        printf ( " would walk away. Then\n" );
        printf ( "      : we have the \"spin_limit\" which shows" );
        printf ( " how many times you are\n" );
        printf ( "      : willing to play roulette regardless of" );
        printf ( " winnings or losses.\n" );
        printf ( "      : Thus, to indicate a starting bankroll of" );
        printf ( " 2000 with a per bet\n" );
        printf ( "      : cost of 25 as well as a reasonable" );
        printf ( " winnings of 500 limited\n" );
        printf ( "      : within 20 spins we would enter :\n" );
        printf ( "      :\n" );
        printf ( "      :     %s 2000 25 500 20\n", argv[0] );
        printf ( "      :\n" );
        printf ( "      : Note: each number must be a positive integer.\n\n" );
        exit( EXIT_FAILURE );
    }

    bankroll_start = (uint32_t) labs( strtol( argv[1], (char **)NULL, 10) );
    bankroll = bankroll_start;

    bet = (uint32_t) labs( strtol( argv[2], (char **)NULL, 10) );
    profit_limit = (uint32_t) labs( strtol( argv[3], (char **)NULL, 10) );
    max_spin = (uint32_t) labs( strtol( argv[4], (char **)NULL, 10) );

    printf ( "      :    Bank roll = %6i\n", bankroll );
    printf ( "      :          bet = %6i\n", bet );
    printf ( "      : profit limit = %6i\n", profit_limit );
    printf ( "      :     Max Spin = %6i\n", max_spin );

    /*
        This is a simple experiment with a fixed betting pattern where we
        cover off 22 numbers on the table.  This is 22/38 = 0.57895 rho_w.

        There are twenty two direct bets at 35:1 payout : 
 
              0, 00,
              4,  5,  6,  7,  8, 11, 14, 16, 17, 18,
             20, 22, 23, 24, 26, 29, 31, 32, 33, 35

        There are twenty split bets at 17:1 payout :

             0 + 00,   4 +  5,   5 +  6,   5 +  8,   4 +  7,
             8 + 11,  11 + 14,  14 + 17,  16 + 17,  17 + 18,
            17 + 20,  20 + 23,  22 + 23,  23 + 24,  23 + 26, 
            26 + 29,  29 + 32,  31 + 32,  32 + 33,  32 + 35

        The above indicates a total of 42 bets with the following
        payouts for each possible number that may occur : 

        number   possible payout                          | total
        -----------------------------------------------------------
          0    |  35:1  +  17:1                           |  52
         00    |  35:1  +  17:1                           |  52
          4    |  35:1  +  17:1  +  17:1                  |  69
          5    |  35:1  +  17:1  +  17:1  +  17:1         |  86
          6    |  35:1  +  17:1                           |  52
          7    |  35:1  +  17:1                           |  52
          8    |  35:1  +  17:1  +  17:1                  |  69
         11    |  35:1  +  17:1  +  17:1                  |  69
         14    |  35:1  +  17:1  +  17:1                  |  69
         16    |  35:1  +  17:1                           |  52
         17    |  35:1  +  17:1  +  17:1 +  17:1  +  17:1 | 103
         18    |  35:1  +  17:1                           |  52
         20    |  35:1  +  17:1  +  17:1                  |  69
         22    |  35:1  +  17:1                           |  52
         23    |  35:1  +  17:1  +  17:1 +  17:1  +  17:1 | 103
         24    |  35:1  +  17:1                           |  52
         26    |  35:1  +  17:1  +  17:1                  |  69
         29    |  35:1  +  17:1  +  17:1                  |  69
         31    |  35:1  +  17:1                           |  52
         32    |  35:1  +  17:1  +  17:1 +  17:1  +  17:1 | 103
         33    |  35:1  +  17:1                           |  52
         35    |  35:1  +  17:1                           |  52

     * Therefore, is we represent a bet chip as "@" then
     * the table looks like this :

    ---------------------------------------------------------------------------+
   /   |     |     |     |     |     |     |     |     |     |     |     |     |
  /    |  3r |  6b |  9r | 12r | 15b | 18r | 21r | 24b | 27r | 30r | 33b | 36r |
 /  @  |     |  @  |     |     |     |  @  |     |  @  |     |     |  @  |     |
+  00  +-----+--@--+-----+-----+-----+--@--+-----+--@--+-----+-----+--@--+-----+
|      |     |     |     |     |     |     |     |     |     |     |     |     |
+---@--+  2b |  5r @  8b @ 11b @ 14r @ 17b @ 20b @ 23r @ 26b @ 29b @ 32r @ 35b |
|      |     |  @  |  @  |  @  |  @  |  @  |  @  |  @  |  @  |  @  |  @  |  @  |
+   0  +-----+--@--+-----+-----+-----+--@--+-----+--@--+-----+-----+--@--+-----+
 \  @  |     |     |     |     |     |     |     |     |     |     |     |     |
  \    |  1r |  4b @  7r | 10b | 13b | 16r | 19r | 22b | 25r | 28b | 31b | 34r |
   \   |     |  @  |  @  |     |     |  @  |     |  @  |     |     |  @  |     |
    ---------------------------------------------------------------------------+


     *  Therefore rho_w = 0.57895  and  rho_l = 1 - rho_w = 0.42105
     *  with rho_l^2 = 0.17729  and  rho_l^3 = 0.07465  and rho_l^4 = 0.03143
     *  where each loss in sequence has a lower probability of happening. 
     *  However each loss has exactly the same rho_l. Therefore a rho_l in
     *  sequence is called rho_lN where N indicates how many losses have 
     *  already occured in sequence.  Thus rho_l0 = 0.42105 since the
     *  probability of a loss in sequence is rho_lN = rho_l0^N.
     *
     *     spin #    total bet      rho_l       total loss    rho_l0^N
     *   ---------------------------------------------------------------------
     *       1           42         0.42105         42        rho_l0 = 0.42105
     *       2           42         0.42105         84        rho_l1 = 0.17729
     *       3           42         0.42105        126        rho_l2 = 0.07465
     *       4           42         0.42105        168        rho_l3 = 0.03143
     *       5           42         0.42105        210        rho_l4 = 0.01323
     *   ---------------------------------------------------------------------
     *
     *   To simplify, one may expect to loss three times in sequence about 
     *   three times in every forty spins.  Because 3/40 = 0.075 which is 
     *   very close to rho_l2.  However in a random environment one may 
     *   only expect that to be two or three or four times in forty spins. 
     *   There is no valid reason NOT to think one may lose five times in
     *   sequence.  That is expected to happen about seven times in 500 spins.
     *
     */

     if ( ( fp = fopen( "/dev/random", "r" ) ) == NULL ) {
         fprintf ( stderr, "%s: can't read read from /dev/random \n", argv[0] );
         fprintf ( stderr, "%s: ABORTING\n", argv[0] );
         perror ( (char *)NULL );
         return ( EXIT_FAILURE );
     }

    /* zero the ball counts */
    for ( i = 0; i < 38; ++i )
        ball[i] = 0;

    iteration_count = 0;
    n_odd = 0;
    n_even = 0;
    n_red = 0;
    n_black = 0;
    n_zero = 0;
    rho = (double) 1.0 / (double) 38.0;

    for ( i = 0; i < 10000000; ++i ) {   /* should be max_spin here */

        j = getc(fp);
        printf ( "     %6i is 0x%02xh = %3i", i, j, j );
        rval = ( (double) j / (double) 256.0 );

        printf ( " and real is %11.8f", rval );
        k = (int) ( rval * (double) 38 );
        printf ( "    k = %2i\n", k );

        /* so long as we didn't end up on a 0 or 00 then
         * we must be even or odd as well as red or black 
         */
        if ( k > 1 ) {
            /* odd or even ? */
            if ( (k-1)%2 )
                n_odd += 1;
            else
                n_even += 1; 

            /* black or red ? */
            if ( colour_data[k-2] > 0 )
                n_red += 1;
            else
                n_black += 1;

        } else
            n_zero += 1;

        ball[k] += 1;
        iteration_count += 1;

    }

    /* print out the number of times the ball landed on each number */
    expected_n = (long) ( rho * iteration_count );

    printf ( "\n\nIterations = %6i\n", iteration_count );
    printf ( "     0 = %6i     %11.8f    %+6i\n", ball[0],
             ( (float) 1.0 * ball[0] / ( (float) 1.0 * iteration_count ) ),
             ball[0] - expected_n );

    printf ( "    00 = %6i     %11.8f    %+6i\n", ball[1],
             ( (float) 1.0 * ball[1] / ( (float) 1.0 * iteration_count ) ),
             ball[1] - expected_n );

    for ( i = 2; i < 38; ++i )
        printf ( "    %2i = %6i     %11.8f    %+6i\n",  ( i - 1 ), ball[i],
             ( (float) 1.0 * ball[i] / ( (float) 1.0 * iteration_count ) ),
             ball[i] - expected_n );

    /* what was the distribution of even and odds ? */
    printf ( "\n   odd = %6i     %11.8f\n", n_odd, 
             ( (float) 1.0 * n_odd   / ( (float) 1.0 * iteration_count ) ) );
    printf ( "  even = %6i     %11.8f\n", n_even, 
             ( (float) 1.0 * n_even  / ( (float) 1.0 * iteration_count ) ) );

    /* what was the distribution of red and black ? */
    printf ( "\n   red = %6i     %11.8f\n", n_red, 
             ( (float) 1.0 * n_red   / ( (float) 1.0 * iteration_count ) ) );
    printf ( " black = %6i     %11.8f\n", n_black, 
             ( (float) 1.0 * n_black / ( (float) 1.0 * iteration_count ) ) );


    return ( EXIT_SUCCESS );

}

