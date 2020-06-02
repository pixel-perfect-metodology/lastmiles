
	.section	".text",#alloc,#execinstr
	.align	8
	.skip	24

	! block 0

	.global	main
	.type	main,#function
main:
	save	%sp,-720,%sp

	! block 1
.L89:
	st	%i0,[%fp+2043]
	stx	%i1,[%fp+2031]

! File roulette.c:
!    1	#include <stdio.h>
!    2	#include <stdlib.h>
!    3	
!    4	int
!    5	main(int argc, char *argv[])
!    6	{
!    7	
!    8	        long int        iteration_count;
!    9	        int             i, j, k, l, fflush_err;
!   10	        int             spin[37];
!   11	        int             spin_sum;
!   12	        int             max_spin_life;
!   13	        int             max_money;
!   14	        int             loop;
!   15	        int             loss_track, max_loss_track;
!   16	        double          rval;
!   17	
!   18	        int             bank_roll, bet, table_max, win;
!   19	        int             third[2];
!   20	
!   21	        char           *prog = argv[0]; /* Name of the program on the command

	ldx	[%fp+2031],%l0
	ldx	[%l0+0],%l0
	stx	%l0,[%fp+1783]

!   22	                                         * line */
!   23	        char            error_buff[255];
!   24	
!   25	        FILE           *fp;
!   26	
!   27	        for (loop = 0; loop < 1; ++loop) {

	mov	0,%l0
	st	%l0,[%fp+1831]

	! block 2
.L94:
.L91:

!   29	                /********************************************/
!   30	                bet = 25;

	mov	25,%i0

!   31	                bank_roll = 10000;

	sethi	%hi(0x2710),%i4
	or	%i4,%lo(0x2710),%i4

!   32	                table_max = 3500;

	mov	3500,%l0
	st	%l0,[%fp+1803]

!   33	                /********************************************/
!   34	                if ((fp = fopen("/dev/random", "r")) == NULL) {

	sethi	%h44(.L96),%l3
	or	%l3,%m44(.L96),%l3
	sllx	%l3,12,%l3
	or	%l3,%l44(.L96),%l3
	add	%l3,16,%o1
	mov	%l3,%o0
	call	fopen
	nop
	stx	%o0,[%fp+1519]
	cmp	%o0,%g0
	bne,pt	%xcc,	.L95
	nop

	! block 3
.L98:

!   35	                        /** There was an error **/
!   36	                        fprintf(stderr, "%s: can't read read from /dev/random \n", prog);

	sethi	%h44(__iob+256),%l1
	or	%l1,%m44(__iob+256),%l1
	sllx	%l1,12,%l1
	or	%l1,%l44(__iob+256),%l1
	add	%l3,24,%o1
	ldx	[%fp+1783],%o2
	mov	%l1,%o0
	call	fprintf
	nop

!   37	                        fprintf(stderr, "%s: ABORTING\n", prog);

	add	%l3,64,%o1
	ldx	[%fp+1783],%o2
	mov	%l1,%o0
	call	fprintf
	nop

!   38	                        perror(error_buff);

	add	%fp,1528,%o0
	call	perror
	nop

!   39	                        exit(1);

	mov	1,%o0
	call	exit
	nop
	ba	.L101
	nop

	! block 4
.L95:

!   40	                } else {
!   41	                        /* zero the spin data */
!   42	                        spin_sum = 0;

	mov	0,%l0
	st	%l0,[%fp+1843]

!   43	                        for (l = 0; l < 38; ++l)

	mov	0,%l0
	st	%l0,[%fp+1999]

	! block 5
.L_y0:
	add	%fp,1847,%l1
.L105:
.L102:

!   44	                                spin[l] = 0;

	ldsw	[%fp+1999],%l2
	sllx	%l2,2,%l0
	st	%g0,[%l1+%l0]
	add	%l2,1,%l0
	cmp	%l0,38
	bl,pt	%icc,	.L102
	st	%l0,[%fp+1999]

	! block 6
.L106:
.L104:

!   46	                        printf("\nBank roll is $ %i\n", bank_roll);

	sethi	%h44(.L107),%o0
	or	%o0,%m44(.L107),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L107),%o0
	mov	%i4,%o1
	call	printf
	nop

!   47	                        loss_track = 0;

	mov	0,%i1

!   48	                        max_loss_track = 0;

	mov	0,%l0
	st	%l0,[%fp+1823]

!   50	                        /********************************************/
!   51	                        /* start the game with a minumum bet on the */
!   52	                        /* first two thirds of the table            */
!   53	                        max_money = bank_roll;

	st	%i4,[%fp+1835]

!   54	                        third[0] = bet;

	mov	%i0,%i3

!   55	                        third[1] = bet;

	mov	%i0,%i2

!   56	                        third[2] = 0;

	mov	0,%i5

!   57	                        bank_roll = bank_roll - (2 * bet);

	sll	%i0,1,%l0
	sub	%i4,%l0,%i4

!   58	                        /********************************************/
!   59	
!   60	                        max_spin_life = 0;

	mov	0,%l0
	st	%l0,[%fp+1839]

!   62	                        for (i = 0; i < 16; ++i) {

	mov	0,%l0
	st	%l0,[%fp+2011]

	! block 7
.L111:
.L108:

!   64	                                max_spin_life = max_spin_life + 1;

	ldsw	[%fp+1839],%l0
	add	%l0,1,%l0
	st	%l0,[%fp+1839]

!   66	                                if (bank_roll > max_money) {

	mov	%i4,%l0
	ldsw	[%fp+1835],%l1
	cmp	%l0,%l1
	ble,pt	%icc,	.L112
	nop

	! block 8
.L113:

!   67	                                        printf("New Peak bank roll $ %i\n", bank_roll);

	sethi	%h44(.L114),%o0
	or	%o0,%m44(.L114),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L114),%o0
	mov	%l0,%o1
	call	printf
	nop

!   68	                                        max_money = bank_roll;

	st	%i4,[%fp+1835]

	! block 9
.L112:

!   69	                                }
!   70	
!   71	                                if ( loss_track > max_loss_track ) {

	mov	%i1,%l0
	ldsw	[%fp+1823],%l1
	cmp	%l0,%l1
	ble,pt	%icc,	.L115
	nop

	! block 10
.L116:

!   72	                                    max_loss_track = loss_track;

	st	%l0,[%fp+1823]

	! block 11
.L115:

!   73	                                }
!   74	
!   75	                                printf("\n--------------------------------");

	sethi	%h44(.L117),%l2
	or	%l2,%m44(.L117),%l2
	sllx	%l2,12,%l2
	or	%l2,%l44(.L117),%l2
	mov	%l2,%o0
	call	printf
	nop

!   76	                                printf("\nBank roll is $ %i\n", bank_roll);

	add	%l2,-56,%o0
	mov	%i4,%o1
	call	printf
	nop

!   77	                                printf("Total Bet %i is ", (third[0] + third[1] + third[2]));

	add	%l2,40,%o0
	add	%i3,%i2,%l1
	add	%l1,%i5,%o1
	call	printf
	nop

!   78	                                if (third[0] > 0)

	mov	%i3,%l0
	cmp	%l0,0
	ble,pt	%icc,	.L119
	nop

	! block 12
.L120:

!   79	                                        printf(" [  %6i  ] ", third[0]);

	add	%l2,64,%o0
	mov	%l0,%o1
	call	printf
	nop
	ba	.L122
	nop

	! block 13
.L119:

!   80	                                else
!   81	                                        printf(" [          ] ");

	add	%l2,80,%o0
	call	printf
	nop

	! block 14
.L122:

!   83	                                if (third[1] > 0)

	mov	%i2,%l0
	cmp	%l0,0
	ble,pt	%icc,	.L124
	nop

	! block 15
.L125:

!   84	                                        printf(" [  %6i  ] ", third[1]);

	sethi	%h44(.L121),%o0
	or	%o0,%m44(.L121),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L121),%o0
	mov	%l0,%o1
	call	printf
	nop
	ba	.L126
	nop

	! block 16
.L124:

!   85	                                else
!   86	                                        printf(" [          ] ");

	sethi	%h44(.L123),%o0
	or	%o0,%m44(.L123),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L123),%o0
	call	printf
	nop

	! block 17
.L126:

!   88	                                if (third[2] > 0)

	mov	%i5,%l0
	cmp	%l0,0
	ble,pt	%icc,	.L127
	nop

	! block 18
.L128:

!   89	                                        printf(" [  %6i  ]\n", third[2]);

	sethi	%h44(.L129),%o0
	or	%o0,%m44(.L129),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L129),%o0
	mov	%l0,%o1
	call	printf
	nop
	ba	.L130
	nop

	! block 19
.L127:

!   90	                                else
!   91	                                        printf(" [          ]\n");

	sethi	%h44(.L131),%o0
	or	%o0,%m44(.L131),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L131),%o0
	call	printf
	nop

	! block 20
.L130:

!   93	                                j = getc(fp);

	ldx	[%fp+1519],%o0
	call	getc
	nop
	st	%o0,[%fp+2007]

!   94	                                /* printf ( "Byte %i is %i", i, j ); */
!   95	
!   96	                                rval = ((double) j / (double) 256);

	st	%o0,[%fp+1515]
	ld	[%fp+1515],%f8
	fitod	%f8,%f10
	sethi	%h44(.L_cseg0),%l0
	or	%l0,%m44(.L_cseg0),%l0
	sllx	%l0,12,%l0
	ldd	[%l0+%l44(.L_cseg0)],%f8
	fdivd	%f10,%f8,%f10
	std	%f10,[%fp+1815]

!   97	                                /* printf ( " and float is %.8f\n", rval ); */
!   98	
!   99	                                k = (int) (rval * (double) 38);

	sethi	%h44(.L_cseg1),%l0
	or	%l0,%m44(.L_cseg1),%l0
	sllx	%l0,12,%l0
	ldd	[%l0+%l44(.L_cseg1)],%f8
	fmuld	%f10,%f8,%f8
	fdtoi	%f8,%f8
	st	%f8,[%fp+1515]
	ldsw	[%fp+1515],%l3
	st	%l3,[%fp+2003]

!  100	                                spin[k] = spin[k] + 1;

	add	%fp,1847,%l2
	sllx	%l3,2,%l0
	add	%l2,%l0,%l1
	ldsw	[%l1+0],%l0
	add	%l0,1,%l0
	st	%l0,[%l1+0]

!  102	                                if (k > 1) {

	cmp	%l3,1
	ble,pt	%icc,	.L132
	nop

	! block 21
.L133:

!  103	                                        printf("Spin is %i\n", (k - 1));

	sethi	%h44(.L134),%o0
	or	%o0,%m44(.L134),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L134),%o0
	sub	%l3,1,%o1
	call	printf
	nop
	ba	.L135
	nop

	! block 22
.L132:

!  104	                                } else {
!  105	                                        if (k > 0)

	cmp	%l3,0
	ble,pt	%icc,	.L136
	nop

	! block 23
.L137:

!  106	                                                printf("Spin is 00\n");

	sethi	%h44(.L138),%o0
	or	%o0,%m44(.L138),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L138),%o0
	call	printf
	nop
	ba	.L139
	nop

	! block 24
.L136:

!  107	                                        else
!  108	                                                printf("Spin is 0\n");

	sethi	%h44(.L140),%o0
	or	%o0,%m44(.L140),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L140),%o0
	call	printf
	nop

	! block 25
.L139:
.L135:

!  109	                                }
!  110	
!  111	                                if (k > 1) {

	ldsw	[%fp+2003],%l0
	cmp	%l0,1
	ble,pt	%icc,	.L141
	nop

	! block 26
.L142:

!  112	                                        if ((k < 14) && (third[0] > 0)) {

	cmp	%l0,14
	bge,pt	%icc,	.L143
	nop

	! block 27
.L144:
	mov	%i3,%l0
	cmp	%l0,0
	ble,pt	%icc,	.L143
	nop

	! block 28
.L145:

!  113	                                                win = (third[0] * 2) - third[1] - third[2];

	sll	%l0,1,%l1
	sub	%l1,%i2,%l1
	sub	%l1,%i5,%i5

!  114	                                                printf("Win $ %i\n", win);

	sethi	%h44(.L146),%o0
	or	%o0,%m44(.L146),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L146),%o0
	mov	%i5,%o1
	call	printf
	nop

!  115	                                                bank_roll = bank_roll + win - bet;

	add	%i4,%i5,%l1
	sub	%l1,%i0,%i4

!  116	                                                third[0] = 0;

	mov	0,%i3

!  117	                                                third[1] = bet;

	mov	%i0,%i2

!  118	                                                third[2] = bet;

	mov	%i0,%i5

!  119	                                                loss_track = 0;


!  120	                                                goto next_spin;

	ba	.L147
	mov	0,%i1

	! block 29
.L143:

!  121	                                        }
!  122	                                        if ((k > 13) && (k < 26) && (third[1] > 0)) {

	ldsw	[%fp+2003],%l0
	cmp	%l0,13
	ble,pt	%icc,	.L148
	nop

	! block 30
.L149:
	cmp	%l0,26
	bge,pt	%icc,	.L148
	nop

	! block 31
.L150:
	mov	%i2,%l0
	cmp	%l0,0
	ble,pt	%icc,	.L148
	nop

	! block 32
.L151:

!  123	                                                win = (third[1] * 2) - third[0] - third[2];

	sll	%l0,1,%l1
	sub	%l1,%i3,%l1
	sub	%l1,%i5,%i5

!  124	                                                printf("Win $ %i\n", win);

	sethi	%h44(.L146),%o0
	or	%o0,%m44(.L146),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L146),%o0
	mov	%i5,%o1
	call	printf
	nop

!  125	                                                bank_roll = bank_roll + win - bet;

	add	%i4,%i5,%l1
	sub	%l1,%i0,%i4

!  126	                                                third[0] = bet;

	mov	%i0,%i3

!  127	                                                third[1] = 0;

	mov	0,%i2

!  128	                                                third[2] = bet;

	mov	%i0,%i5

!  129	                                                loss_track = 0;


!  130	                                                goto next_spin;

	ba	.L147
	mov	0,%i1

	! block 33
.L148:

!  131	                                        }
!  132	                                        if ((k > 25) && (third[2] > 0)) {

	ldsw	[%fp+2003],%l0
	cmp	%l0,25
	ble,pt	%icc,	.L152
	nop

	! block 34
.L153:
	mov	%i5,%l0
	cmp	%l0,0
	ble,pt	%icc,	.L152
	nop

	! block 35
.L154:

!  133	                                                win = (third[2] * 2) - third[0] - third[1];

	sll	%l0,1,%l1
	sub	%l1,%i3,%l1
	sub	%l1,%i2,%i5

!  134	                                                printf("Win $ %i\n", win);

	sethi	%h44(.L146),%o0
	or	%o0,%m44(.L146),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L146),%o0
	mov	%i5,%o1
	call	printf
	nop

!  135	                                                bank_roll = bank_roll + win - bet;

	add	%i4,%i5,%l1
	sub	%l1,%i0,%i4

!  136	                                                third[0] = bet;

	mov	%i0,%i3

!  137	                                                third[1] = bet;

	mov	%i0,%i2

!  138	                                                third[2] = 0;

	mov	0,%i5

!  139	                                                loss_track = 0;


!  140	                                                goto next_spin;

	ba	.L147
	mov	0,%i1

	! block 36
.L152:

!  141	                                        }
!  142	                                        /* this is a loss from here on in */
!  143	                                        printf("Loss of $ %i\n", (third[0] + third[1] + third[2]));

	sethi	%h44(.L155),%l7
	or	%l7,%m44(.L155),%l7
	sllx	%l7,12,%l7
	or	%l7,%l44(.L155),%l7
	add	%i3,%i2,%l1
	add	%l1,%i5,%o1
	mov	%l7,%o0
	call	printf
	nop

!  144	                                        loss_track = loss_track + 1;

	add	%i1,1,%i1

!  145	                                        if (bank_roll >= ((third[0] * 3) + (third[1] * 3) + (third[2] * 3))) {

	mov	%i4,%l4
	sll	%i3,1,%l0
	add	%i3,%l0,%l5
	sll	%i2,1,%l0
	add	%i2,%l0,%l6
	add	%l5,%l6,%l3
	sll	%i5,1,%l0
	add	%i5,%l0,%l1
	add	%l3,%l1,%l2
	cmp	%l4,%l2
	bl,pt	%icc,	.L156
	nop

	! block 37
.L157:

!  146	                                                bank_roll = bank_roll - ((third[0] * 3) + (third[1] * 3) + (third[2] * 3));

	sub	%l4,%l2,%i4

!  147	                                                third[0] = (third[0] * 3);

	mov	%l5,%i3

!  148	                                                third[1] = (third[1] * 3);

	mov	%l6,%i2

!  149	                                                third[2] = (third[2] * 3);

	ba	.L158
	mov	%l1,%i5

	! block 38
.L156:

!  150	                                        } else {
!  151	                                                printf("You probably can not continue further.\n");

	add	%l7,16,%o0
	call	printf
	nop

!  152	                                                goto done;

	ba	.L160
	nop

	! block 39
.L158:
.L147:

!  153	                                        }
!  154	
!  155	                        next_spin:
!  156	                                        printf("\n");

	sethi	%h44(.L161),%o0
	or	%o0,%m44(.L161),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(.L161),%o0
	call	printf
	nop
	ba	.L162
	nop

	! block 40
.L141:

!  157	                                } else {
!  158	                                        printf("Total Loss of $ %i\n", (third[0] + third[1] + third[2]));

	sethi	%h44(.L163),%l7
	or	%l7,%m44(.L163),%l7
	sllx	%l7,12,%l7
	or	%l7,%l44(.L163),%l7
	add	%i3,%i2,%l1
	add	%l1,%i5,%o1
	mov	%l7,%o0
	call	printf
	nop

!  159	                                        if (bank_roll >= ((third[0] * 3) + (third[1] * 3) + (third[2] * 3))) {

	mov	%i4,%l4
	sll	%i3,1,%l0
	add	%i3,%l0,%l5
	sll	%i2,1,%l0
	add	%i2,%l0,%l6
	add	%l5,%l6,%l3
	sll	%i5,1,%l0
	add	%i5,%l0,%l1
	add	%l3,%l1,%l2
	cmp	%l4,%l2
	bl,pt	%icc,	.L164
	nop

	! block 41
.L165:

!  160	                                                bank_roll = bank_roll - ((third[0] * 3) + (third[1] * 3) + (third[2] * 3));

	sub	%l4,%l2,%i4

!  161	                                                third[0] = (third[0] * 3);

	mov	%l5,%i3

!  162	                                                third[1] = (third[1] * 3);

	mov	%l6,%i2

!  163	                                                third[2] = (third[2] * 3);

	mov	%l1,%i5

!  164	                                                loss_track = loss_track + 1;

	ba	.L166
	add	%i1,1,%i1

	! block 42
.L164:

!  165	                                        } else {
!  166	                                                printf("You probably can not continue further.\n");

	add	%l7,-48,%o0
	call	printf
	nop

!  167	                                                goto done;

	ba	.L160
	nop

	! block 43
.L166:
.L162:
	ldsw	[%fp+2011],%l0
	add	%l0,1,%l0
	cmp	%l0,16
	bl,pt	%icc,	.L108
	st	%l0,[%fp+2011]

	! block 44
.L167:
.L110:
.L160:

!  168	                                        }
!  169	                                }
!  170	                                /********************************************/
!  171	
!  172	                        }       /* end of for i */
!  173	
!  174	        done:
!  175	
!  176	                        fflush_err = fflush(fp);

	ldx	[%fp+1519],%o0
	call	fflush
	nop
	mov	%o0,%l0
	st	%l0,[%fp+1995]

!  177	                        if (fflush_err != 0) {

	cmp	%l0,0
	be,pt	%icc,	.L168
	nop

	! block 45
.L169:

!  178	                                fprintf(stderr, "fflush error %i", fflush_err);

	sethi	%h44(__iob+256),%o0
	or	%o0,%m44(__iob+256),%o0
	sllx	%o0,12,%o0
	or	%o0,%l44(__iob+256),%o0
	sethi	%h44(.L170),%o1
	or	%o1,%m44(.L170),%o1
	sllx	%o1,12,%o1
	or	%o1,%l44(.L170),%o1
	mov	%l0,%o2
	call	fprintf
	nop

!  179	                                exit(1);

	mov	1,%o0
	call	exit
	nop

	! block 46
.L168:

!  180	                        }
!  181	                        fclose(fp);     /* close the file and flush buffers */

	ldx	[%fp+1519],%o0
	call	fclose
	nop

!  183	                        /*
!  184	                         * printf ( "spin[ 0 ] = %i\n", spin[0] ); printf (
!  185	                         * "spin[ 00 ] = %i\n", spin[1] ); spin_sum = spin[0]
!  186	                         * + spin[1]; for ( l = 2; l < 38 ; ++l ) { spin_sum
!  187	                         * = spin_sum + spin[l]; printf ( "spin[ %i ] =
!  188	                         * %i\n", (l-1), spin[l] ); } printf ( "      sum  =
!  189	                         * %i\n", spin_sum );
!  190	                         */
!  191	                        printf(" life was %i\n", max_spin_life);

	sethi	%h44(.L171),%l0
	or	%l0,%m44(.L171),%l0
	sllx	%l0,12,%l0
	or	%l0,%l44(.L171),%l0
	ldsw	[%fp+1839],%o1
	mov	%l0,%o0
	call	printf
	nop

!  192	                        printf(" max sequential losses was %i\n", max_loss_track );

	add	%l0,16,%o0
	ldsw	[%fp+1823],%o1
	call	printf
	nop

!  193	                        printf(" max money $ %i\n", max_money);

	add	%l0,48,%o0
	ldsw	[%fp+1835],%o1
	call	printf
	nop

!  194	                        printf("\nfinal Bank roll is $ %i\n", bank_roll);

	add	%l0,72,%o0
	mov	%i4,%o1
	call	printf
	nop

	! block 47
.L101:
	ldsw	[%fp+1831],%l0
	add	%l0,1,%l0
	cmp	%l0,1
	bl,pt	%icc,	.L91
	st	%l0,[%fp+1831]

	! block 48
.L175:
.L93:

!  195	                }
!  196	        }
!  197	
!  198	        return (0);

	mov	0,%l0
	st	%l0,[%fp+2027]
	mov	0,%i0
	jmp	%i7+8
	restore

	! block 49
.L88:
	mov	0,%i0
	jmp	%i7+8
	restore
	.size	main,(.-main)
	.align	8

	.section	".rodata1",#alloc
	.align	8
.L96:
	.ascii	"/dev/random\000"
	.skip	4
	.type	.L96,#object
	.size	.L96,16
	.align	8
.L97:
	.ascii	"r\000"
	.skip	6
	.type	.L97,#object
	.size	.L97,8
	.align	8
.L99:
	.ascii	"%s: can't read read from /dev/random \n\000"
	.skip	1
	.type	.L99,#object
	.size	.L99,40
	.align	8
.L100:
	.ascii	"%s: ABORTING\n\000"
	.skip	2
	.type	.L100,#object
	.size	.L100,16
	.align	8
.L107:
	.ascii	"\nBank roll is $ %i\n\000"
	.skip	4
	.type	.L107,#object
	.size	.L107,24
	.align	8
.L114:
	.ascii	"New Peak bank roll $ %i\n\000"
	.skip	7
	.type	.L114,#object
	.size	.L114,32
	.align	8
.L117:
	.ascii	"\n--------------------------------\000"
	.skip	6
	.type	.L117,#object
	.size	.L117,40
	.align	8
.L118:
	.ascii	"Total Bet %i is \000"
	.skip	7
	.type	.L118,#object
	.size	.L118,24
	.align	8
.L121:
	.ascii	" [  %6i  ] \000"
	.skip	4
	.type	.L121,#object
	.size	.L121,16
	.align	8
.L123:
	.ascii	" [          ] \000"
	.skip	1
	.type	.L123,#object
	.size	.L123,16
	.align	8
.L129:
	.ascii	" [  %6i  ]\n\000"
	.skip	4
	.type	.L129,#object
	.size	.L129,16
	.align	8
.L131:
	.ascii	" [          ]\n\000"
	.skip	1
	.type	.L131,#object
	.size	.L131,16
	.align	8
.L134:
	.ascii	"Spin is %i\n\000"
	.skip	4
	.type	.L134,#object
	.size	.L134,16
	.align	8
.L138:
	.ascii	"Spin is 00\n\000"
	.skip	4
	.type	.L138,#object
	.size	.L138,16
	.align	8
.L140:
	.ascii	"Spin is 0\n\000"
	.skip	5
	.type	.L140,#object
	.size	.L140,16
	.align	8
.L146:
	.ascii	"Win $ %i\n\000"
	.skip	6
	.type	.L146,#object
	.size	.L146,16
	.align	8
.L155:
	.ascii	"Loss of $ %i\n\000"
	.skip	2
	.type	.L155,#object
	.size	.L155,16
	.align	8
.L159:
	.ascii	"You probably can not continue further.\n\000"
	.type	.L159,#object
	.size	.L159,40
	.align	8
.L161:
	.ascii	"\n\000"
	.skip	6
	.type	.L161,#object
	.size	.L161,8
	.align	8
.L163:
	.ascii	"Total Loss of $ %i\n\000"
	.skip	4
	.type	.L163,#object
	.size	.L163,24
	.align	8
.L170:
	.ascii	"fflush error %i\000"
	.type	.L170,#object
	.size	.L170,16
	.align	8
.L171:
	.ascii	" life was %i\n\000"
	.skip	2
	.type	.L171,#object
	.size	.L171,16
	.align	8
.L172:
	.ascii	" max sequential losses was %i\n\000"
	.skip	1
	.type	.L172,#object
	.size	.L172,32
	.align	8
.L173:
	.ascii	" max money $ %i\n\000"
	.skip	7
	.type	.L173,#object
	.size	.L173,24
	.align	8
.L174:
	.ascii	"\nfinal Bank roll is $ %i\n\000"
	.type	.L174,#object
	.size	.L174,26

	.section	".rodata",#alloc
	.align	8
.L_cseg0:
	.xword	0x4070000000000000LL
	.type	.L_cseg0,#object
	.size	.L_cseg0,8
	.align	8
.L_cseg1:
	.xword	0x4043000000000000LL
	.type	.L_cseg1,#object
	.size	.L_cseg1,8

	.section	".bss",#alloc,#write
Bbss.bss:
	.skip	0
	.type	Bbss.bss,#object
	.size	Bbss.bss,0

	.section	".data",#alloc,#write
Ddata.data:
	.skip	0
	.type	Ddata.data,#object
	.size	Ddata.data,0

	.section	".rodata",#alloc
Drodata.rodata:
	.skip	0
	.type	Drodata.rodata,#object
	.size	Drodata.rodata,0

	.file	"roulette.c"
	.ident	"@(#)stdio.h	1.78	99/12/08 SMI"
	.ident	"@(#)stdio_iso.h	1.2	99/10/25 SMI"
	.ident	"@(#)feature_tests.h	1.18	99/07/26 SMI"
	.ident	"@(#)isa_defs.h	1.20	99/05/04 SMI"
	.ident	"@(#)va_list.h	1.12	99/05/04 SMI"
	.ident	"@(#)stdio_tag.h	1.3	98/04/20 SMI"
	.ident	"@(#)stdio_impl.h	1.8	99/06/10 SMI"
	.ident	"@(#)stdlib.h	1.47	99/11/03 SMI"
	.ident	"@(#)stdlib_iso.h	1.3	02/10/10 SMI"
	.ident	"acomp: Sun C 5.8 Patch 121015-02 2006/03/29"

	.global	__fsr_init_value
__fsr_init_value = 0x0
!  Begin sdCreateSection : .debug_info
!  Section Info: link_name/strtab=, entsize=0x1, adralign=0x1, flags=0x0
!  Section Data Blocks:
!   reloc[0]: knd=2, off=14, siz=8, lab1=.debug_abbrev, lab2=, loff=0
!   reloc[1]: knd=2, off=255, siz=8, lab1=.debug_line, lab2=, loff=0
	.section ".debug_info"
	.byte 0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00
	.byte 0x00,0x00,0x00,0xfc,0x00,0x02
	.uaxword %section_symbol(".debug_abbrev")
	.byte 0x08,0x01,0x00,0x00,0x00,0x00,0x00,0x00
	.byte 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
	.byte 0x00,0x04,0x72,0x6f,0x75,0x6c,0x65,0x74
	.byte 0x74,0x65,0x2e,0x63,0x00,0x0c,0x2f,0x65
	.byte 0x78,0x70,0x6f,0x72,0x74,0x2f,0x6d,0x65
	.byte 0x64,0x75,0x73,0x61,0x2f,0x64,0x63,0x6c
	.byte 0x61,0x72,0x6b,0x65,0x2f,0x70,0x67,0x6d
	.byte 0x2f,0x43,0x2f,0x72,0x61,0x6e,0x64,0x6f
	.byte 0x6d,0x2f,0x72,0x6f,0x75,0x6c,0x65,0x74
	.byte 0x74,0x65,0x00,0x20,0x2f,0x6f,0x70,0x74
	.byte 0x2f,0x53,0x55,0x4e,0x57,0x73,0x70,0x72
	.byte 0x6f,0x2f,0x70,0x72,0x6f,0x64,0x2f,0x62
	.byte 0x69,0x6e,0x2f,0x63,0x63,0x20,0x2d,0x78
	.byte 0x73,0x74,0x72,0x63,0x6f,0x6e,0x73,0x74
	.byte 0x20,0x2d,0x78,0x69,0x6c,0x64,0x6f,0x66
	.byte 0x66,0x20,0x2d,0x78,0x61,0x72,0x63,0x68
	.byte 0x3d,0x76,0x39,0x61,0x20,0x2d,0x78,0x6c
	.byte 0x69,0x62,0x6d,0x69,0x6c,0x20,0x2d,0x58
	.byte 0x63,0x20,0x2d,0x53,0x20,0x20,0x72,0x6f
	.byte 0x75,0x6c,0x65,0x74,0x74,0x65,0x2e,0x63
	.byte 0x00,0x58,0x61,0x3b,0x52,0x3d,0x53,0x75
	.byte 0x6e,0x20,0x43,0x20,0x35,0x2e,0x38,0x20
	.byte 0x50,0x61,0x74,0x63,0x68,0x20,0x31,0x32
	.byte 0x31,0x30,0x31,0x35,0x2d,0x30,0x32,0x20
	.byte 0x32,0x30,0x30,0x36,0x2f,0x30,0x33,0x2f
	.byte 0x32,0x39,0x3b,0x62,0x61,0x63,0x6b,0x65
	.byte 0x6e,0x64,0x3b,0x72,0x61,0x77,0x3b,0x63
	.byte 0x64,0x3b,0x00,0x44,0x42,0x47,0x5f,0x47
	.byte 0x45,0x4e,0x20,0x35,0x2e,0x30,0x2e,0x38
	.byte 0x00
	.uaxword %section_symbol(".debug_line")
	.byte 0x00
!  End sdCreateSection
!  Begin sdCreateSection : .debug_line
!  Section Info: link_name/strtab=, entsize=0x1, adralign=0x1, flags=0x0
!  Section Data Blocks:
	.section ".debug_line"
	.byte 0xff,0xff,0xff,0xff,0x00,0x00,0x00,0x00
	.byte 0x00,0x00,0x00,0x55,0x00,0x02,0x00,0x00
	.byte 0x00,0x00,0x00,0x00,0x00,0x4b,0x04,0x00
	.byte 0xff,0x04,0x0a,0x00,0x01,0x01,0x01,0x01
	.byte 0x00,0x00,0x00,0x01,0x2f,0x65,0x78,0x70
	.byte 0x6f,0x72,0x74,0x2f,0x6d,0x65,0x64,0x75
	.byte 0x73,0x61,0x2f,0x64,0x63,0x6c,0x61,0x72
	.byte 0x6b,0x65,0x2f,0x70,0x67,0x6d,0x2f,0x43
	.byte 0x2f,0x72,0x61,0x6e,0x64,0x6f,0x6d,0x2f
	.byte 0x72,0x6f,0x75,0x6c,0x65,0x74,0x74,0x65
	.byte 0x00,0x00,0x72,0x6f,0x75,0x6c,0x65,0x74
	.byte 0x74,0x65,0x2e,0x63,0x00,0x01,0x00,0x00
	.byte 0x00
!  End sdCreateSection
!  Begin sdCreateSection : .debug_abbrev
!  Section Info: link_name/strtab=, entsize=0x1, adralign=0x1, flags=0x0
!  Section Data Blocks:
	.section ".debug_abbrev"
	.byte 0x01,0x11,0x00,0x11,0x01,0x12,0x01,0x03
	.byte 0x08,0x13,0x0b,0x1b,0x08,0x85,0x44,0x08
	.byte 0x87,0x44,0x08,0x25,0x08,0x10,0x07,0x00
	.byte 0x00,0x00
!  End sdCreateSection
