#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char *argv[])
{

	long int	iteration_count;
	int	i, j, k, l, fflush_err;
	int	spin[37];
	int	spin_sum;
	int	max_spin_life;
	int	max_money;
	int	loop;
	int	loss_track, max_loss_track;
	double	rval;

	int	bank_roll, bet, table_max, win;
	int	third[2];

	char	*prog = argv[0]; /* Name of the program on the command
	                                         * line */
	char	error_buff[255];

	FILE           * fp;

	for (loop = 0; loop < 1; ++loop) {

		/********************************************/
		bet = 25;
		bank_roll = 10000;
		table_max = 3500;
		/********************************************/
		if ((fp = fopen("/dev/random", "r")) == NULL) {
			/** There was an error **/
			fprintf(stderr, "%s: can't read read from /dev/random \n", prog);
			fprintf(stderr, "%s: ABORTING\n", prog);
			perror(error_buff);
			exit(1);
		} else {
			/* zero the spin data */
			spin_sum = 0;
			for (l = 0; l < 38; ++l)
				spin[l] = 0;

			printf("\nBank roll is $ %i\n", bank_roll);
			loss_track = 0;
			max_loss_track = 0;

			/********************************************/
			/* start the game with a minumum bet on the */
			/* first two thirds of the table            */
			max_money = bank_roll;
			third[0] = bet;
			third[1] = bet;
			third[2] = 0;
			bank_roll = bank_roll - (2 * bet);
			/********************************************/

			max_spin_life = 0;

			for (i = 0; i < 16; ++i) {

				max_spin_life = max_spin_life + 1;

				if (bank_roll > max_money) {
					printf("New Peak bank roll $ %i\n", bank_roll);
					max_money = bank_roll;
				}

				if ( loss_track > max_loss_track ) {
					max_loss_track = loss_track;
				}

				printf("\n--------------------------------");
				printf("\nBank roll is $ %i\n", bank_roll);
				printf("Total Bet %i is ", (third[0] + third[1] + third[2]));
				if (third[0] > 0)
					printf(" [  %6i  ] ", third[0]);
					else
					printf(" [          ] ");

				if (third[1] > 0)
					printf(" [  %6i  ] ", third[1]);
					else
					printf(" [          ] ");

				if (third[2] > 0)
					printf(" [  %6i  ]\n", third[2]);
					else
					printf(" [          ]\n");

				j = getc(fp);
				/* printf ( "Byte %i is %i", i, j ); */

				rval = ((double) j / (double) 256);
				/* printf ( " and float is %.8f\n", rval ); */

				k = (int) (rval * (double) 38);
				spin[k] = spin[k] + 1;

				if (k > 1) {
					printf("Spin is %i\n", (k - 1));
				} else {
					if (k > 0)
						printf("Spin is 00\n");
						else
						printf("Spin is 0\n");
				}

				if (k > 1) {
					if ((k < 14) && (third[0] > 0)) {
						win = (third[0] * 2) - third[1] - third[2];
						printf("Win $ %i\n", win);
						bank_roll = bank_roll + win - bet;
						third[0] = 0;
						third[1] = bet;
						third[2] = bet;
						loss_track = 0;
						goto next_spin;
					}
					if ((k > 13) && (k < 26) && (third[1] > 0)) {
						win = (third[1] * 2) - third[0] - third[2];
						printf("Win $ %i\n", win);
						bank_roll = bank_roll + win - bet;
						third[0] = bet;
						third[1] = 0;
						third[2] = bet;
						loss_track = 0;
						goto next_spin;
					}
					if ((k > 25) && (third[2] > 0)) {
						win = (third[2] * 2) - third[0] - third[1];
						printf("Win $ %i\n", win);
						bank_roll = bank_roll + win - bet;
						third[0] = bet;
						third[1] = bet;
						third[2] = 0;
						loss_track = 0;
						goto next_spin;
					}
					/* this is a loss from here on in */
					printf("Loss of $ %i\n", (third[0] + third[1] + third[2]));
					loss_track = loss_track + 1;
					if (bank_roll >= ((third[0] * 3) + (third[1] * 3) + (third[2] * 3))) {
						bank_roll = bank_roll - ((third[0] * 3) + (third[1] * 3) + (third[2] * 3));
						third[0] = (third[0] * 3);
						third[1] = (third[1] * 3);
						third[2] = (third[2] * 3);
					} else {
						printf("You probably can not continue further.\n");
						goto done;
					}

next_spin:
					printf("\n");
				} else {
					printf("Total Loss of $ %i\n", (third[0] + third[1] + third[2]));
					if (bank_roll >= ((third[0] * 3) + (third[1] * 3) + (third[2] * 3))) {
						bank_roll = bank_roll - ((third[0] * 3) + (third[1] * 3) + (third[2] * 3));
						third[0] = (third[0] * 3);
						third[1] = (third[1] * 3);
						third[2] = (third[2] * 3);
						loss_track = loss_track + 1;
					} else {
						printf("You probably can not continue further.\n");
						goto done;
					}
				}
				/********************************************/

			}       /* end of for i */

done:

			fflush_err = fflush(fp);
			if (fflush_err != 0) {
				fprintf(stderr, "fflush error %i", fflush_err);
				exit(1);
			}
			fclose(fp);     /* close the file and flush buffers */

			/*
			                         * printf ( "spin[ 0 ] = %i\n", spin[0] ); printf (
			                         * "spin[ 00 ] = %i\n", spin[1] ); spin_sum = spin[0]
			                         * + spin[1]; for ( l = 2; l < 38 ; ++l ) { spin_sum
			                         * = spin_sum + spin[l]; printf ( "spin[ %i ] =
			                         * %i\n", (l-1), spin[l] ); } printf ( "      sum  =
			                         * %i\n", spin_sum );
			                         */
			printf(" life was %i\n", max_spin_life);
			printf(" max sequential losses was %i\n", max_loss_track );
			printf(" max money $ %i\n", max_money);
			printf("\nfinal Bank roll is $ %i\n", bank_roll);
		}
	}

	return (0);

}                               /* End of main */
