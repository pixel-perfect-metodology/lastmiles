
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

#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include "mandelbrot.h" 

/* This will be a dispatched POSIX pthread that shall receive
 * a point to a struct of type thread_parm_t.  Then we shall
 * pull out the values needed from that struct and simply
 * compute the mandelbrot height for each coordinate in a
 * given vbox region on the screen */
void *mbrot_vbox_pthread(void *recv_parm)
{
    thread_parm_t *p = (thread_parm_t *)recv_parm;
    double win_x, win_y, x_prime, y_prime;
    int mand_x_pix, mand_y_pix, vbox_ll_x, vbox_ll_y;
    int mand_y_pix_start, mand_y_pix_stop;

   /* point c belongs to the Mandelbrot set if and only if
    * the magnitude of the f(c) <= 2.0 */
    uint32_t height;
    double zr, zi, tmp_r, tmp_i, mag;

    /* lets come up with an imaginary axis start_i and stop_i
     * based on this thread t_num */
    mand_y_pix_start = ( p->vbox_h / p->t_total ) * p->t_num;
    /* actually the stop line is one less than this next thing */
    mand_y_pix_stop = mand_y_pix_start + ( p->vbox_h / p->t_total );

    fprintf (stderr,"[ t%02i ] : %-2i -> %-2i\n", p->t_num, mand_y_pix_start, mand_y_pix_stop - 1);

    for ( mand_y_pix = mand_y_pix_start; mand_y_pix < mand_y_pix_stop; mand_y_pix++ ) {

        /* lower left corner of this threads little rectangle */
        vbox_ll_y = p->vbox_y * p->vbox_h + mand_y_pix;

        for ( mand_x_pix = 0; mand_x_pix < p->vbox_w; mand_x_pix++ ) {

            /* we compute from the lower left corner of the on screen
             * vbox going left to right and upwards along the positive
             * imaginary axis. */
            vbox_ll_x = p->vbox_x * p->vbox_w + mand_x_pix;

            win_x = ( ( ( 1.0 * vbox_ll_x ) / p->eff_width ) * 2.0 - 1.0 ) + 0.0;
            win_y = ( -1.0 * ( ( ( 1.0 * ( p->eff_height - vbox_ll_y ) ) / p->eff_height ) * 2.0 - 1.0 ) ) + 0.0;

            x_prime = p->obs_x_width * win_x / 2.0;
            y_prime = p->obs_y_height * win_y / 2.0;

            x_prime = x_prime + p->r_translate;
            y_prime = y_prime + p->i_translate;

            height = 0;
            zr = 0.0;
            zi = 0.0;
            mag = 0.0;

            while ( ( height < ( p->bail_out ) ) && ( mag <= 4.0 ) ) {
                tmp_r = ( zr * zr ) - ( zi * zi );
                tmp_i = ( zr * zi ) + ( zr * zi );
                zr = tmp_r + x_prime;
                zi = tmp_i + y_prime;
                mag = zr * zr + zi * zi;
                height += 1;
            }

            (*(p->v))[p->vbox_x][p->vbox_y][mand_x_pix][mand_y_pix] = height;

        }
    }

    p->ret_val = 0;

    return ( NULL );

}

