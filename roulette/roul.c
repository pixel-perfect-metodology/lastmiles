#define _POSIX_SOURCE 1
#define BANKROLL 2000
#define BET 25
#define PROFIT 1000

#include <stdio.h>
#include <stdlib.h>

void print_bets ();

int third[2];

int
main(int argc, char *argv[])
{

    /* Note : the american style double zero wheel has numbers 
     *        arranged thus : 
     *         0 28  9 26 30 11  7 20 32 17  5 22 34 15  3 24 36 13  1
     *        00 27 10 25 29 12  8 19 31 18  6 21 33 16  4 23 35 14  2
     *
     * The layout looks somewhat like this : 

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

    long int    iteration_count;
    int    i, j, k, l, fflush_err;

    /* the ball[] array is a record of the number of times that the 
     * roulette wheel ball lands on a given number with ball[0] being
     * the single 0 and ball[1] being the double 00.
     */
    int ball[37];

    int    spin_sum, max_spin_life, max_money, min_money, walk_away;
    int    loop, loss_track, max_loss_track, max_spin;
    double rval;

    int    bank_roll, bankroll_start, bet, table_max, win;
    int    profit_limit;

    int    n_even, n_odd, n_red, n_black, colour_flag;

    char   *prog = argv[0]; 
    char   error_buff[255];
    FILE   * fp;

    if ( argc < 5 ) {
        printf ( "usage : %s bankroll bet profit_limit spin_limit\n", prog );
        printf ( "      : example data 2000 25 500 20\n" );
        exit( EXIT_FAILURE );
    } 

    bankroll_start = atoi ( argv[1] );
    bet = atoi ( argv[2] );
    profit_limit = atoi ( argv[3] );
    max_spin = atoi ( argv[4] );

    printf ( "      :    Bank roll = %5i\n", bankroll_start );
    printf ( "      :          bet = %5i\n", bet );
    printf ( "      : profit limit = %5i\n", profit_limit );
    printf ( "      :     Max Spin = %5i\n", max_spin );

    for ( loop = 0; loop < 1; ++loop ) {

        bank_roll = bankroll_start;
        table_max = 3500;
        walk_away = 0;

        if ( ( fp = fopen( "/dev/random", "r" ) ) == NULL ) {
            /** There was an error **/
            fprintf( stderr, "%s: can't read read from /dev/random \n", prog );
            fprintf( stderr, "%s: ABORTING\n", prog );
            perror( error_buff );
            exit( EXIT_FAILURE );
        } else {
            /* zero the spin data */
            spin_sum = 0;
            for ( l = 0; l < 38; ++l )
                ball[l] = 0;

            printf("\n Bank roll is %6i\n", bank_roll);
            loss_track = 0;
            max_loss_track = 0;

            /********************************************/
            /* start the game with a minumum bet on the */
            /* first two thirds of the table            */
            max_money = bank_roll;
            min_money = bank_roll;
            third[0] = bet;
            third[1] = bet;
            third[2] = 0;
            bank_roll = bank_roll - ( 2 * bet );
            printf (" Spend %6i on bets.\n", ( 2 * bet ) );
            print_bets ();
            printf(" Bank roll %6i\n", bank_roll);
            /********************************************/

            max_spin_life = 0;

            for ( i = 0; i < max_spin; ++i ) {

                max_spin_life = max_spin_life + 1;

                if ( bank_roll > max_money ) {
                    /* printf(" Peak roll %6i\n", bank_roll); */
                    max_money = bank_roll;
                }

                if ( min_money > bank_roll ) {
                    min_money = bank_roll;
                }

                if ( loss_track > max_loss_track ) {
                    max_loss_track = loss_track;
                }
                printf(" %4i :-------------------------------------------------------------\n", i );
                print_bets ();
                j = getc(fp);
                /* printf ( "Byte %i is %i", i, j ); */
                rval = ( (double) j / (double) 256);
                /* printf ( " and float is %.8f\n", rval ); */

                k = (int) ( rval * (double) 38 );

                if ( ( k > 1 ) && ( (k-1)%2 ) )
                    n_odd += 1;
                else
                    n_even += 1;

                /* colour_flag shall be 1 for red and 0 for black */
                if ( ( k > 1 ) &&
                     ( ( k ==  1+1 ) || ( k ==  3+1 ) || ( k ==  5+1 ) ||
                       ( k ==  7+1 ) || ( k ==  9+1 ) || ( k == 12+1 ) ||
                       ( k == 14+1 ) || ( k == 16+1 ) || ( k == 18+1 ) ||
                       ( k == 19+1 ) || ( k == 21+1 ) || ( k == 23+1 ) ||
                       ( k == 25+1 ) || ( k == 27+1 ) || ( k == 30+1 ) ||
                       ( k == 32+1 ) || ( k == 34+1 ) || ( k == 36+1 )
                     ) ) {
                    n_red += 1; 
                    colour_flag = 1;
                } else {
                    n_black += 1;
                    colour_flag = 0;
                }

                ball[k] = ball[k] + 1;
                win = 0;

                if ( k < 2 ) {
                    if ( k == 1 )
                        printf(" BALL is 00\n");
                    else
                        printf(" BALL is  0\n");
                } else {
                    printf(" BALL is %2i ", (k - 1));
                    if ( colour_flag == 0 ) printf( "black " );
                        else printf( "  red " );
                    if ( (k-1)%2 ) printf( " odd\n" );
                        else printf( "even\n" );

                    if ( ( k < 14 ) && ( third[0] > 0 ) ) {
                        win = ( third[0] * 3 );
                        printf ( " Win %4i\n", win );
                        bank_roll = bank_roll + win;
                        printf ( " Bank roll %6i\n", bank_roll );
                        printf ( " Spend %6i on bets.\n", ( 2 * bet ) );
                        third[0] = 0;
                        third[1] = bet;
                        third[2] = bet;
                        bank_roll = bank_roll - ( 2 * bet );
                        printf ( " Bank roll %6i\n", bank_roll );
                        loss_track = 0;
                    }
                    if ( ( k > 13 ) && ( k < 26 ) && ( third[1] > 0 ) ) {
                        win = ( third[1] * 3 );
                        printf ( " Win %4i\n", win );
                        bank_roll = bank_roll + win;
                        printf ( " Bank roll %6i\n", bank_roll );
                        printf ( " Spend %6i on bets.\n", ( 2 * bet ) );
                        third[0] = bet;
                        third[1] = 0;
                        third[2] = bet;
                        bank_roll = bank_roll - ( 2 * bet );
                        printf ( " Bank roll %6i\n", bank_roll );
                        loss_track = 0;
                    }
                    if ( ( k > 25 ) && ( third[2] > 0 ) ) {
                        win = ( third[2] * 3 );
                        printf ( " Win %4i\n", win );
                        bank_roll = bank_roll + win;
                        printf ( " Bank roll %6i\n", bank_roll );
                        printf ( " Spend %6i on bets.\n", ( 2 * bet ) );
                        third[0] = bet;
                        third[1] = bet;
                        third[2] = 0;
                        bank_roll = bank_roll - ( 2 * bet );
                        printf ( " Bank roll %6i\n", bank_roll );
                        loss_track = 0;
                    }
                }

                if ( win == 0 ) {
                    printf(" Lose %5i\n", (third[0] + third[1] + third[2]));
                    if (bank_roll >= ((third[0] * 3) + (third[1] * 3) + (third[2] * 3))) {
                        bank_roll = bank_roll - ((third[0] * 3) + (third[1] * 3) + (third[2] * 3));
                        printf ( " Spend %6i on bets.\n", 
                                    ( third[0] * 3 ) 
                                  + ( third[1] * 3 ) 
                                  + ( third[2] * 3 ) );
                        printf(" Bank roll %6i\n", bank_roll);
                        third[0] = (third[0] * 3);
                        third[1] = (third[1] * 3);
                        third[2] = (third[2] * 3);
                        loss_track = loss_track + 1;
                    } else {
                        printf("You probably can not continue further.\n");
                        goto done;
                    }
                }

               if ( bank_roll >= ( bankroll_start + profit_limit ) ) {
                   walk_away = 1;
                   goto done;
               }
            }

done:

            fflush_err = fflush(fp);
            if (fflush_err != 0) {
                fprintf(stderr, "fflush error %i", fflush_err);
                exit( EXIT_FAILURE );
            }
            fclose(fp);     /* close the file and flush buffers */

            if ( bank_roll > max_money ) {
                printf(" Peak roll %6i\n", bank_roll );
                max_money = bank_roll;
            }

            printf( "\nFINAL : %6i : %3i spins : %6i max : %6i min : %3i l_seq : %i\n",
                  bank_roll, max_spin_life, max_money, min_money, max_loss_track, walk_away );
        }
    }

    return ( EXIT_SUCCESS );

}

void print_bets () { 

    printf ( " Bet %4i on ", ( third[0] + third[1] + third[2] ) );

    if ( third[0] > 0 )
        printf ( " [  %6i  ] ", third[0] );
    else
        printf ( " [          ] " );

    if ( third[1] > 0 )
        printf ( " [  %6i  ] ", third[1] );
    else
        printf ( " [          ] " );

    if ( third[2] > 0 )
        printf ( " [  %6i  ]\n", third[2] );
    else
        printf ( " [          ]\n" );

}

