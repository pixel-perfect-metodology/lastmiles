
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

#include <X11/Xlib.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <time.h>
#include <locale.h>
#include <unistd.h>
#include <math.h>
#include <errno.h>

#include <pthread.h>

Window create_borderless_topwin(Display *dsp,
                         int width, int height,
                         int x, int y, int bg_color);

GC create_gc(Display *dsp, Window win);

int X_error_handler(Display *dsp, XErrorEvent *errevt);

uint64_t timediff( struct timespec st, struct timespec en );

int sysinfo(void);

uint32_t mandle_col ( uint8_t height );

uint32_t mbrot( double c_r, double c_i, uint32_t bail_out );

uint32_t mbrot_subpixel ( Display *d, Window *w, GC *g,
                          int mand_x_pix, int mand_y_pix,
                          double x_prime, double y_prime,
                          double pixel_width, double pixel_height,
                          uint32_t mand_bail );

/* local defs where 1044 pixels is more or less full screen
 * and 660 pixels square fits into a neat 720p res OBS setup */
#define WIN_WIDTH 1044
#define WIN_HEIGHT 1044
#define OFFSET 10

typedef struct cplex {
    double r, i;
} cplex_type;

int main(int argc, char*argv[])
{
    /* our display and window and graphics context */
    Display *dsp;
    Window win, win2, win3;
    GC gc, gc2, gc3;
    Colormap screen_colormap;
    XEvent event;
    Font fixed_font, type_font;

    /* a very few colours */
    XColor red, green, blue, yellow, cyan, magenta;
    XColor cornflowerblue, royal_blue, very_dark_grey;
    XColor mandlebrot;

    Status retcode0;  /* not really needed for trivial stuff */
    int retcode1, status;     /* curious what some funcs return */

    /* setup mouse x and y */
    int mouse_x = -1, mouse_y = -1;
    int invert_mouse_x,  invert_mouse_y;

    /* these next five are just mouse button counters where the
     * roll_up and roll_dn are mouse wheel events */
    int button, left_count, mid_count, right_count, roll_up, roll_dn;
    left_count = 0;
    mid_count = 0;
    right_count = 0;
    roll_up = 0;
    roll_dn = 0;

    uint64_t t_delta;

    struct timespec t0, t1, now_time;
    struct timespec soln_t0, soln_t1;

    /* some primordial vars */
    int disp_width, disp_height, width, height;
    int conn_num, screen_num, depth;
    int j, k, p, q, offset_x, offset_y;
    int lx, ly, ux, uy, px, py;
    int gc2_x, gc2_y;
    int eff_width, eff_height, vbox_w, vbox_h;
    double obs_x_width, obs_y_height;
    double magnify, real_translate, imag_translate;
    double x_prime, y_prime;
    double pixel_width, pixel_height;

    /* use the vbox lower left coords as reference */
    int vbox_ll_x, vbox_ll_y;

    /* we need to track when a vbox has been computed
     * as well as displayed via libX11. For now we just
     * don't want to recompute the same region over and
     * over and over. */
    int vbox_flag[16][16];

    uint32_t mand_height;
    uint32_t mand_bail = 1024;

    cplex_type mand_tmp, mand_z, mand_c;
    int mand_x_pix, mand_y_pix;
    double mand_mag;

    /* These are the initial and normalized mouse fp64 values
     * from within the viewport. */
    double win_x, win_y;

    /* Also we finally have use for the little box grid that we
     * lay out and thus we will need the box coordinates */
    int vbox_x, vbox_y;

    /* small general purpose char buffer */
    char *buf = calloc((size_t)128,sizeof(unsigned char));

    char *disp_name = NULL;

    setlocale( LC_ALL, "C" );

    /* Get the REALTIME_CLOCK time in a timespec struct */
    if ( clock_gettime( CLOCK_REALTIME, &now_time ) == -1 ) {
        /* We could not get the clock. Bail out. */
        fprintf(stderr,"ERROR : could not attain CLOCK_REALTIME\n");
        return(EXIT_FAILURE);
    } else {
        /* call srand48() with the sub-second time data */
        srand48( (long) now_time.tv_nsec );
    }
    sysinfo();

    /* test region 1
     *    real_translate = -7.368164062500e-01;
     *    imag_translate = -1.818847656250e-01;
     *
     * test region 2
     *    real_translate = -1.769241333008;
     *    imag_translate = -5.691528320312e-02;
     *
     * test region 3
     *    magnify = pow( 2.0, 16.0);
     *    real_translate = -7.622470855713e-01;
     *    imag_translate = -8.939456939698e-02;
     *    real_translate = -1.609520912171e-01;
     *    imag_translate = -1.038573741913;
     */

    /* scale and translate */
    magnify = pow( 2.0, 20.0);
    real_translate = -1.609513163567e-01;
    imag_translate = -1.038571417332e+00;
    printf("translate = ( %-+18.12e , %-+18.12e )\n",
                                      real_translate, imag_translate );

    printf("  magnify = %g\n", magnify );

    obs_x_width = 4.0 / magnify;
    obs_y_height = 4.0 / magnify;

    pixel_width = obs_x_width / width;
    pixel_height = obs_y_height / height;

    /* ensure we start with clear vbox flags */
    for ( p=0; p<16; p++ )
        for ( q=0; q<16; q++ )
            vbox_flag[p][q] = 0;

    width = WIN_WIDTH;
    height = WIN_HEIGHT;
    printf("\n X11 : ");
    printf("default width=%4i height=%4i\n", width, height);

    XSetErrorHandler(X_error_handler);

    /* should work with a null display name */
    dsp = XOpenDisplay(disp_name);
    if (dsp == NULL) {
        fprintf(stderr, "%s: no X server?? '%s'\n",
            argv[0], disp_name);
        exit(EXIT_FAILURE);
    }
    conn_num = XConnectionNumber(dsp);
    printf("     : connection number %i\n", conn_num);

    screen_num = DefaultScreen(dsp);
    printf("     : screen number %i\n", screen_num);

    depth = XDefaultDepth(dsp,screen_num);
    printf("     : default depth is %i\n", depth);

    fixed_font = XLoadFont(dsp, "fixed");

    /* really we need to get a list of the available fonts and then
     * use one that should work in the correct size
     */
    type_font = XLoadFont(dsp, "lucidasanstypewriter-10");

    disp_width = DisplayWidth(dsp, screen_num);
    disp_height = DisplayHeight(dsp, screen_num);

    printf("     : display seems to be %i wide and %i high.\n",
                                   disp_width, disp_height);

    if ( ( disp_width < width ) || ( disp_height < height ) ) {
        fprintf(stderr, "ERROR: screen is too small\n\n");
        exit(EXIT_FAILURE);
    }

    /* hard coded screen offset */
    offset_x = 20;
    offset_y = 20;

    printf("     : offset x=%i y=%i\n", offset_x, offset_y);

    /* Our primary plotting windows is pale grey on the screen */
    win = create_borderless_topwin(dsp, width, height,
                                        offset_x, offset_y,
                                        0x0f0f0f);
    gc = create_gc(dsp, win);

    /* create a smaller darker window to the right */
    unsigned long gc2_bg = 0x101000;
    win2 = create_borderless_topwin(dsp, 400, 330, 1070, 740, gc2_bg );
    gc2 = create_gc(dsp, win2);
    XSetBackground(dsp, gc2, gc2_bg);

    /* create another small window below that */
    unsigned long gc3_bg = 0x001010;
    win3 = create_borderless_topwin(dsp, 440, 330, 1470, 740, gc3_bg );
    gc3 = create_gc(dsp, win2);
    XSetBackground(dsp, gc3, gc3_bg);

    XSync(dsp, False);

    screen_colormap = XDefaultColormap(dsp, DefaultScreen(dsp));
    if (XAllocNamedColor(dsp,
                         screen_colormap,
                         "red", &red, &red) == 0) {
        fprintf(stderr, "XAllocNamedColor - no red color?\n");
        exit(EXIT_FAILURE);
    }
    if (XAllocNamedColor(dsp,
                         screen_colormap,
                         "green", &green, &green) == 0) {
        fprintf(stderr, "XAllocNamedColor - red works but green no??\n");
        exit(EXIT_FAILURE);
    }
    if (XAllocNamedColor(dsp,
                         screen_colormap,
                         "blue", &blue, &blue) == 0) {
        fprintf(stderr, "XAllocNamedColor - red and green okay but blue??\n");
        exit(EXIT_FAILURE);
    }
    if (XAllocNamedColor(dsp,
                         screen_colormap,
                         "yellow", &yellow, &yellow) == 0) {
        fprintf(stderr, "XAllocNamedColor - yellow bork bork bork!\n");
        exit(EXIT_FAILURE);
    }

    if (XAllocNamedColor(dsp,
                         screen_colormap,
                         "cyan", &cyan, &cyan) == 0) {
        fprintf(stderr, "XAllocNamedColor - cyan bork bork bork!\n");
        exit(EXIT_FAILURE);
    }

    if (XAllocNamedColor(dsp,
                         screen_colormap,
                         "magenta", &magenta, &magenta) == 0) {
        fprintf(stderr, "XAllocNamedColor - magenta bork bork!\n");
        exit(EXIT_FAILURE);
    }

    /* cornflowerblue is #6495ED */
    if (XAllocNamedColor(dsp,
                         screen_colormap,
                         "cornflowerblue",
                         &cornflowerblue, &cornflowerblue) == 0) {
        fprintf(stderr, "XAllocNamedColor - cornflowerblue fails.\n");
        exit(EXIT_FAILURE);
    }

    /* request Royal Blue which should be #4169E1 however we
     * will get whatever teh hardware can map closest to the
     * request */
    royal_blue.flags= DoRed | DoGreen | DoBlue;
    royal_blue.red = 0x4100;
    royal_blue.green = 0x6900;
    royal_blue.blue = 0xe100;
    if ( XAllocColor(dsp, screen_colormap, &royal_blue) == 0 ) {
        fprintf(stderr, "XAllocColor - royal_blue fails.\n");
        exit(EXIT_FAILURE);
    }

    /* We need an inner grid which in our main plot window
     * which should be a subtle very very dark grey.
     * Here we manually define the rgb components using 16bit
     * values and then create the new color */
    very_dark_grey.flags= DoRed | DoGreen | DoBlue;
    very_dark_grey.red = 0x1f00;
    very_dark_grey.green = 0x1f00;
    very_dark_grey.blue = 0x1f00;
    if ( XAllocColor(dsp, screen_colormap, &very_dark_grey) == 0 ) {
        fprintf(stderr, "XAllocColor - very_dark_grey fails.\n");
        exit(EXIT_FAILURE);
    }

    /* this is a hack color data value that we will abuse later
     * inside the mouse Xevent loop */
    mandlebrot.flags= DoRed | DoGreen | DoBlue;
    /* some dummy values */
    mandlebrot.green = 0x0000;
    mandlebrot.blue  = 0x0000;
    mandlebrot.red   = 0xff00;

    if ( XAllocColor(dsp, screen_colormap, &mandlebrot) == 0 ) {
        fprintf(stderr, "XAllocColor - gee .. mandlebrot fail.\n");
        exit(EXIT_FAILURE);
    }

    /* main plot window yellow pixel at each corner 5 pixels indent */
    XSetForeground(dsp, gc, yellow.pixel);
    XDrawPoint(dsp, win, gc, 5, 5);
    XDrawPoint(dsp, win, gc, 5, height - 5);
    XDrawPoint(dsp, win, gc, width - 5, 5);
    XDrawPoint(dsp, win, gc, width - 5, height - 5);

    /* draw a blue box inside the second window */
    XSetForeground(dsp, gc2, blue.pixel);

    XSetLineAttributes(dsp, gc2, 1, LineSolid,
                                    CapButt,
                                    JoinMiter);

    XDrawRectangle(dsp, win2, gc2, 5, 5, 390, 320);
    XSetForeground(dsp, gc2, cyan.pixel);

    /* a little window to plot the vbox data into with a 3x3 grid
     * for each pixel we sample.  This shall be the 64x64 actual
     * vbox region with room to plot each of the 3x3 samples and
     * we also need room for the one pixel borders.
     */
    XDrawRectangle(dsp, win2, gc2, 10, 10, 202, 202);
    XSetForeground(dsp, gc2, red.pixel);

    /* draw a blue box inside the third window */
    XSetForeground(dsp, gc3, blue.pixel);
    XSetLineAttributes(dsp, gc3, 1, LineSolid,
                                    CapButt,
                                    JoinMiter);

    XDrawRectangle(dsp, win3, gc3, 5, 5, 430, 320);

    /* set our graph box inside by OFFSET pixels
     *
     * reuse the offset_foo vars from above as we do not
     * need them for window location anymore. So we can
     * use them as interior offset distances for our plot.
     *
     * TODO : maybe make these a unique name and not just
     * redefine the values we used earlier. Maybe. */
    offset_x = 10;
    offset_y = 10;

    /* upper left point */
    ux = offset_x;
    uy = offset_y;

    /* lower right point */
    lx = width - offset_x;
    ly = height - offset_y;

    /* therefore we have effective box width and height */
    eff_width = lx - ux;
    eff_height = ly - uy;

    printf("     : eff_width = %5i    eff_height = %5i\n\n",
                           eff_width, eff_height);
    printf("--------------------------------------------\n");

    XSetLineAttributes(dsp, gc, 1, LineSolid,
                                   CapButt,
                                   JoinMiter);

    XSetForeground(dsp, gc, WhitePixel(dsp, screen_num));
    XSetForeground(dsp, gc2, green.pixel);
    XSetFont(dsp, gc2, type_font);
    XSetFont(dsp, gc3, type_font);

    /****************************************************************
     *
     * The viewport is made up of a neat grid of 16 x 16 little box
     * areas and we can lay down a lightly colored dashed lines to
     * indicate where they are. We may as well refer to these little
     * boxes as view box regions. Starting from the lower left at
     * vbox [0] [0] upwards to the upper most right corner which
     * we can call vbox [15] [15].
     *
     * Each of these vbox elements has a height and width in the
     * on screen pixels of :
     *
     *     vbox_w = eff_width/16
     *
     *     vbox_h = eff_height/16
     *
     * These may come in handy later to identify where the user has
     * clicked and to perhaps identify a small region that can be
     * computed without the burder of computing the entire viewport.
     *
     ****************************************************************/
    vbox_w = eff_width/16;
    vbox_h = eff_height/16;

    for ( j=offset_x + vbox_w; j<lx; j+=vbox_w ){
        XDrawLine(dsp, win, gc, j, 8, j, 12);
        XDrawLine(dsp, win, gc, j, height - 8, j, height - 12);
    }
    XFlush(dsp);

    /* vertical minor tic marks at every 16th of the interior viewport
     * drawing area */
    for ( j = offset_y + vbox_h; j < ly; j += vbox_h ){
        XDrawLine(dsp, win, gc, 8, j, 12, j);
        XDrawLine(dsp, win, gc, width - 8, j, width - 12, j);
    }
    XFlush(dsp);

    /* now we use the very dark grey color we created */
    XSetForeground(dsp,gc,very_dark_grey.pixel);

    /* draw the vertical lines */
    for ( j= offset_x + vbox_w; j<lx; j+=vbox_w ){
        XDrawLine(dsp, win, gc, j, 13, j, height-13);
    }

    /* draw the horizontal lines */
    for ( j = offset_y + vbox_h; j<ly; j+=vbox_h ){
        XDrawLine(dsp, win, gc, 13, j, width-13, j);
    }

    /* gc3 green text as default */
    XSetForeground(dsp, gc3, green.pixel);

    /* royal blue border around the main viewport */
    XSetForeground(dsp, gc, royal_blue.pixel);
    XDrawLine(dsp, win, gc, 10, 10, width - 10, 10);
    XDrawLine(dsp, win, gc, width - 10, 10, width - 10, height - 10);
    XDrawLine(dsp, win, gc, width - 10, height - 10, 10, height - 10);
    XDrawLine(dsp, win, gc, 10, height - 10, 10, 10);

    XFlush(dsp);

    /* TODO : at the moment the only events we are trapping are
     * the mouse buttons but in the future we will want to redraw
     * and re-expose the window if there other event types */

    XGrabPointer(dsp, win, False, ButtonPressMask, GrabModeAsync,
                           GrabModeAsync, None, None, CurrentTime);

    XSelectInput(dsp, win, ButtonPressMask);

    /* some initial time data before anyone clicks anything */
    clock_gettime( CLOCK_MONOTONIC, &t0 );
    clock_gettime( CLOCK_MONOTONIC, &t1 );
    t_delta = timediff( t0, t1 );
    /* this t_delta is a baseline offset value wherein we at least
     * know how long the clock_gettime takes. Mostly. */

    sprintf(buf,"[0000] tdelta = %16lld nsec", t_delta);
    XDrawImageString( dsp, win3, gc3, 10, 20, buf, strlen(buf));

    /* plot some points on the grid that we created */
    XSetForeground(dsp, gc, yellow.pixel);
    XDrawPoint(dsp, win, gc, 5, 5);
    /* TODO at some point check for why we are fetching the x and y
     * values over and over and over inside the switch-case */
    while(1){

        XNextEvent(dsp,&event);

        switch(event.type){
            case ButtonPress:
                switch(event.xbutton.button){
                    case Button1: /* left mouse button */
                        mouse_x=event.xbutton.x;
                        mouse_y=event.xbutton.y;
                        button=Button1;
                        left_count += 1;
                        break;

                    case Button2: /* middle mouse scroll button */
                        mouse_x=event.xbutton.x;
                        mouse_y=event.xbutton.y;
                        button=Button2;
                        mid_count += 1;
                        break;

                    case Button3: /* right mouse button */
                        mouse_x=event.xbutton.x;
                        mouse_y=event.xbutton.y;
                        button=Button3;
                        right_count += 1;
                        break;

                    case Button4: /* mouse scroll wheel up */
                        mouse_x=event.xbutton.x;
                        mouse_y=event.xbutton.y;
                        button=Button4;
                        roll_up += 1;
                        break;

                    case Button5: /* mouse scroll wheel down */
                        mouse_x=event.xbutton.x;
                        mouse_y=event.xbutton.y;
                        button=Button5;
                        roll_dn += 1;
                        break;

                    default:
                        break;
                }
            break;
        default:
            break;
        }

        /* adjustment of one or two pixels */
        mouse_x = mouse_x - 1;
        mouse_y = mouse_y - 2;

        if ( button == Button1 ){

            /* printf("leftclick"); */

            if (    ( mouse_x >=  offset_x ) && ( mouse_y >= offset_y )
                 && ( mouse_x < ( eff_width + offset_x ) )
                 && ( mouse_y < ( eff_height + offset_y ) ) ) {

                /* we are inside the primary window plotting region
                 * so lets try to create floating point values for
                 * the coordinates selected. We start with just a
                 * normalized value from zero to one. */

                win_x = ( 1.0 * ( mouse_x - offset_x ) ) / eff_width;
                win_y = ( 1.0 * ( eff_height - mouse_y + offset_y ) ) / eff_height;

                /* lets try to invert the y axis */
                invert_mouse_x = mouse_x - offset_x;
                invert_mouse_y = eff_height - mouse_y + offset_y;
                sprintf(buf,"inv  [ %4i , %4i ]  ", invert_mouse_x, invert_mouse_y );

                XSetForeground(dsp, gc2, green.pixel);
                XDrawImageString( dsp, win2, gc2, 10, 230, buf, strlen(buf));

                sprintf(buf,"fp64( %-10.8e , %-10.8e )", win_x, win_y );
                XDrawImageString( dsp, win2, gc2, 10, 250, buf, strlen(buf));

                /* a more useful value is the vbox[] coordinates for
                 * each of the 16x16 grid we previously laid out */
                vbox_x = ( mouse_x - offset_x ) / vbox_w;
                vbox_y = ( eff_height - mouse_y + offset_y ) / vbox_h;
                sprintf(buf,"vbox  [ %03i , %03i ]", vbox_x, vbox_y );
                XDrawImageString( dsp, win2, gc2, 10, 270, buf, strlen(buf));
                fprintf(stderr,"%s\n", buf);

                /* Offset the floating point values such that the
                 * center point shall be ( 0.0, 0.0 ) */
                win_x = win_x * 2.0 - 1.0;
                win_y = win_y * 2.0 - 1.0;

                XSetForeground(dsp, gc2, cornflowerblue.pixel);
                sprintf(buf,"fp64( %-+10.8e , %-+10.8e )  ", win_x, win_y );
                XDrawImageString( dsp, win2, gc2, 10, 290, buf, strlen(buf));

                /* At this moment we have normalized values for a
                 * location within the observation viewport. We can
                 * scale those values by half of the viewport width
                 * and height to get actual x_prime and y_prime
                 * values.
                 *
                 * All of the above allows us to compute a starting
                 * point on the observation plane
                 *
                 * Note the x_prime = obs_x_width * win_x / 2.0
                 *          y_prime = obs_y_height * win_y / 2.0
                 */

                x_prime = obs_x_width * win_x / 2.0;
                y_prime = obs_y_height * win_y / 2.0;

                x_prime = x_prime + real_translate;
                y_prime = y_prime + imag_translate;

                XSetForeground(dsp, gc3, red.pixel);
                sprintf(buf,"c = ( %-10.8e , %-10.8e )  ", x_prime, y_prime );
                /* fprintf(stderr,"\n%s\n",buf); */
                fprintf(stderr,"c = ( %-+18.12e , %-+18.12e )\n", x_prime, y_prime );
                XDrawImageString( dsp, win3, gc3, 10, 80, buf, strlen(buf));
                XSetForeground(dsp, gc3, cyan.pixel);

                /* time the computation of the intercepts etc */
                clock_gettime( CLOCK_MONOTONIC, &soln_t0 );

                for ( mand_y_pix = 0; mand_y_pix < vbox_h; mand_y_pix++ ) {
                    vbox_ll_y = vbox_y * vbox_h + mand_y_pix;
                    for ( mand_x_pix = 0; mand_x_pix < vbox_w; mand_x_pix++ ) {
                        vbox_ll_x = vbox_x * vbox_w + mand_x_pix;
                        /*
                        sprintf(buf,"vbox_ll = %-6i , %-6i", vbox_ll_x, vbox_ll_y);
                        XDrawImageString( dsp, win3, gc3, 10, 40, buf, strlen(buf));
                        */

                        win_x = ( ( ( 1.0 * vbox_ll_x ) / eff_width ) * 2.0 - 1.0 ) + 0.0;
                        win_y = ( -1.0 * ( ( ( 1.0 * ( eff_height - vbox_ll_y ) ) / eff_height ) * 2.0 - 1.0 ) ) + 0.0;

                        x_prime = obs_x_width * win_x / 2.0;
                        y_prime = obs_y_height * win_y / 2.0;

                        x_prime = x_prime + real_translate;
                        y_prime = y_prime + imag_translate;

                        /*
                        sprintf(buf,"c = ( %-10.8e , %-10.8e )", x_prime, y_prime );
                        fprintf(stderr,"%s    ",buf);
                        XDrawImageString( dsp, win3, gc3, 10, 60, buf, strlen(buf));
                        */

                        mand_height = mbrot( x_prime, y_prime, mand_bail );

                        /*
                        sprintf(buf,"mand_height = %-8i", mand_height);
                        fprintf(stderr,"%s\n",buf);
                        XDrawImageString( dsp, win3, gc3, 10, 100, buf, strlen(buf));
                        */

                        if ( mand_height == mand_bail ) {
                            XSetForeground(dsp, gc, (unsigned long)0 );
                            XSetForeground(dsp, gc2, (unsigned long)0 );
                        } else {
                            mandlebrot.pixel = (unsigned long)mandle_col ( (uint8_t)(mand_height & 0xff) );
                            XSetForeground(dsp, gc, mandlebrot.pixel);
                            XSetForeground(dsp, gc2, mandlebrot.pixel);
                        }

                        XDrawPoint(dsp, win, gc,
                                   vbox_ll_x + offset_x,
                                   ( eff_height - vbox_ll_y + offset_y ) );

                        /* TODO subpixel average */
                        mbrot_subpixel ( dsp, &win2, &gc2,
                                         mand_x_pix, mand_y_pix,
                                         x_prime, y_prime,
                                         pixel_width, pixel_height,
                                         mand_bail );

                        /*
                        gc2_x = 16 + ( 3 * mand_x_pix );
                        gc2_y = 13 + ( 192 - ( 3 * mand_y_pix ) );
                        XDrawPoint( dsp, win2, gc2, gc2_x, gc2_y );
                        XDrawPoint( dsp, win2, gc2, gc2_x + 1, gc2_y );
                        XDrawPoint( dsp, win2, gc2, gc2_x + 2, gc2_y );
                        XDrawPoint( dsp, win2, gc2, gc2_x, gc2_y + 1 );
                        XDrawPoint( dsp, win2, gc2, gc2_x + 1, gc2_y + 1 );
                        XDrawPoint( dsp, win2, gc2, gc2_x + 2, gc2_y + 1 );
                        XDrawPoint( dsp, win2, gc2, gc2_x, gc2_y + 2 );
                        XDrawPoint( dsp, win2, gc2, gc2_x + 1, gc2_y + 2 );
                        XDrawPoint( dsp, win2, gc2, gc2_x + 2, gc2_y + 2 );
                        */
                    }
                }
                /* thanks to mosh this was on the X11 clipboard
                 *
                 * https://www.beeradvocate.com/beer/profile/813/7451/
                 *
                 * regardless lets track that we actualyl did display
                 * this vbox data already.
                 *
                 * Also .. what was that tripl kermeliet beer ?
                 *
                 */

                vbox_flag[vbox_x][vbox_y] = 1;

                clock_gettime( CLOCK_MONOTONIC, &soln_t1 );

                t_delta = timediff( soln_t0, soln_t1 );
                sprintf(buf,"[soln] = %16lld nsec", t_delta);
                fprintf(stderr,"%s\n",buf);
                XSetForeground(dsp, gc3, green.pixel);
                XDrawImageString( dsp, win3, gc3, 10, 290,
                                  buf, strlen(buf));

            }

        } else if ( button == Button2 ) {

            if (    ( mouse_x >=  offset_x ) && ( mouse_y >= offset_y )
                 && ( mouse_x < ( eff_width + offset_x ) )
                 && ( mouse_y < ( eff_height + offset_y ) ) ) {

                /* we are inside the primary window plotting region */
                win_x = ( 1.0 * ( mouse_x - offset_x ) ) / eff_width;
                win_y = ( 1.0 * ( eff_height - mouse_y + offset_y ) ) / eff_height;

                /* invert the y axis */
                invert_mouse_x = mouse_x - offset_x;
                invert_mouse_y = eff_height - mouse_y + offset_y;
                sprintf(buf,"inv  [ %4i , %4i ]  ", invert_mouse_x, invert_mouse_y );

                XSetForeground(dsp, gc2, green.pixel);
                XDrawImageString( dsp, win2, gc2, 10, 230, buf, strlen(buf));

                sprintf(buf,"fp64( %-10.8e , %-10.8e )", win_x, win_y );
                XDrawImageString( dsp, win2, gc2, 10, 250, buf, strlen(buf));

                /* vbox[] coordinates for the 16x16 grid */
                vbox_x = ( mouse_x - offset_x ) / vbox_w;
                vbox_y = ( eff_height - mouse_y + offset_y ) / vbox_h;
                sprintf(buf,"vbox  [ %03i , %03i ]", vbox_x, vbox_y );
                XDrawImageString( dsp, win2, gc2, 10, 270, buf, strlen(buf));
                fprintf(stderr,"%s\n", buf);

                /* Offset the floating point values such that the
                 * center point shall be ( 0.0, 0.0 ) */
                win_x = win_x * 2.0 - 1.0;
                win_y = win_y * 2.0 - 1.0;

                XSetForeground(dsp, gc2, cornflowerblue.pixel);
                sprintf(buf,"fp64( %-+10.8e , %-+10.8e )  ", win_x, win_y );
                XDrawImageString( dsp, win2, gc2, 10, 290, buf, strlen(buf));

                x_prime = obs_x_width * win_x / 2.0;
                y_prime = obs_y_height * win_y / 2.0;

                /* translation */
                x_prime = x_prime + real_translate;
                y_prime = y_prime + imag_translate;

                XSetForeground(dsp, gc3, red.pixel);
                sprintf(buf,"c = ( %-10.8e , %-10.8e )  ", x_prime, y_prime );
                /* fprintf(stderr,"\n%s\n",buf); */
                fprintf(stderr,"c = ( %-+18.12e , %-+18.12e )\n", x_prime, y_prime );
                XDrawImageString( dsp, win3, gc3, 10, 80, buf, strlen(buf));
                XSetForeground(dsp, gc3, cyan.pixel);

                clock_gettime( CLOCK_MONOTONIC, &soln_t0 );
                /* here we loop over the vbox coords */
                for ( vbox_y = 0; vbox_y < 16; vbox_y++ ) {
                    for ( vbox_x = 0; vbox_x < 16; vbox_x++ ) {
                        if ( vbox_flag[vbox_x][vbox_y] == 0 ) {
                            for ( mand_y_pix = 0; mand_y_pix < vbox_h; mand_y_pix++ ) {
                                vbox_ll_y = vbox_y * vbox_h + mand_y_pix;
                                for ( mand_x_pix = 0; mand_x_pix < vbox_w; mand_x_pix++ ) {
                                    vbox_ll_x = vbox_x * vbox_w + mand_x_pix;

                                    win_x = ( ( ( 1.0 * vbox_ll_x )
                                                / eff_width ) * 2.0 - 1.0 ) + 0.0;

                                    win_y = ( -1.0 *
                                              ( (
                                                   ( 1.0 * ( eff_height - vbox_ll_y ) ) / eff_height
                                                ) * 2.0 - 1.0
                                              ) ) + 0.0;

                                    x_prime = obs_x_width * win_x / 2.0;
                                    y_prime = obs_y_height * win_y / 2.0;

                                    x_prime = x_prime + real_translate;
                                    y_prime = y_prime + imag_translate;

                                    mand_height = mbrot( x_prime, y_prime, mand_bail );
                                    if ( mand_height == mand_bail ) {
                                        XSetForeground(dsp, gc, (unsigned long)0 );
                                    } else {
                                        mandlebrot.pixel = (unsigned long)mandle_col ( (uint8_t)(mand_height & 0xff) );
                                        XSetForeground(dsp, gc, mandlebrot.pixel);
                                    }
                                    XDrawPoint(dsp, win, gc, vbox_ll_x + offset_x, ( eff_height - vbox_ll_y + offset_y ) );

                                }
                            }
                            vbox_flag[vbox_x][vbox_y] = 1;
                        }
                    }
                }
            } /* inside main plot area check */

            XSetForeground(dsp, gc, yellow.pixel);
            clock_gettime( CLOCK_MONOTONIC, &soln_t1 );
            t_delta = timediff( soln_t0, soln_t1 );
            sprintf(buf,"[mand] = %16lld nsec", t_delta);
            fprintf(stderr,"%s\n\n",buf);
            XSetForeground(dsp, gc2, red.pixel);
            XDrawImageString( dsp, win2, gc2, 10, 310,
                                       buf, strlen(buf));

        } else if ( button == Button3 ) {

            printf("right click\n");
            clock_gettime( CLOCK_MONOTONIC, &t1 );
            t_delta = timediff( t0, t1 );

            sprintf(buf,"[%04i] tdelta = %16lld nsec",
                                            right_count, t_delta);

            XDrawImageString( dsp, win3, gc3, 10, 20,
                               buf, strlen(buf));

            t0.tv_sec = t1.tv_sec;
            t0.tv_nsec = t1.tv_nsec;
            /* If a 200ms right double click anywhere then quit */
            if ( t_delta < 200000000 ) {
                printf("\n\n");
                /* If we allocate memory for any purpose whatsoever
                 * then we had better free() it. */
                break;
            }

        } else if ( button == Button4 ) {

            /* TODO note that a mouse wheel event being used here to
             * track observation plane position will result in all
             * data being redrawn. */
            printf("roll up\n");

        } else if ( button == Button5 ) {

            printf("roll down\n");

        } else {

            printf("\n ??? unknown button ???\n");

        }

        printf("click at %d %d \n", mouse_x, mouse_y);

    }

    XCloseDisplay(dsp);

    printf("\n");

    free(buf);
    return EXIT_SUCCESS;
}

uint32_t mbrot( double c_r, double c_i, uint32_t bail_out )
{

    /* point c belongs to the Mandelbrot set if and only if
     * the magnitude of the f(c) <= 2.0 */
    uint32_t height = 0;
    double zr = 0.0;
    double zi = 0.0;
    double tmp_r, tmp_i;
    double mag = 0.0;

    while ( ( height < bail_out ) && ( mag < 2.0 ) ) {
        tmp_r = ( zr * zr ) - ( zi * zi );
        tmp_i = ( zr * zi ) + ( zr * zi );
        zr = tmp_r + c_r;
        zi = tmp_i + c_i;
        mag = sqrt( zr * zr + zi * zi );
        height += 1;
    }

    return ( height );

}

uint32_t mbrot_subpixel ( Display *d, Window *w, GC *g,
                          int mand_x_pix, int mand_y_pix,
                          double x_prime, double y_prime,
                          double pixel_width, double pixel_height,
                          uint32_t mand_bail )
{

    int j, k, gc2_x, gc2_y;
    uint32_t sub_pixel_height, sub_pixel[3][3];
    uint32_t red, green, blue, color_avg;
    double x, y, delta_x, delta_y;

    delta_x = pixel_width / 3.0;
    delta_y = pixel_height / 3.0;

    gc2_x = 16 + ( 3 * mand_x_pix );
    gc2_y = 13 + ( 192 - ( 3 * mand_y_pix ) );

    red = 0;
    green = 0;
    blue = 0;
    color_avg = 0;

    for ( j=0; j<3; j++ ) {
        for ( k=0; k<3; k++ ) {
            x = ( (double)( j - 1 ) * delta_x ) + x_prime;
            y = ( (double)( k - 1 ) * delta_y ) + y_prime;
            sub_pixel_height = mbrot( x, y, mand_bail );

            if ( sub_pixel_height == mand_bail ) {
                sub_pixel[j][k] = 0;
            } else {
                sub_pixel[j][k] = mandle_col( (uint8_t)(sub_pixel_height & 0xff) );
            }

            red   += ( sub_pixel[j][k] & 0xff0000 ) >> 16;
            green += ( sub_pixel[j][k] & 0x00ff00 ) >> 8;
            blue  += ( sub_pixel[j][k] & 0x0000ff );

            XSetForeground( d, *g, (unsigned long)sub_pixel[j][k] );
            XDrawPoint( d, *w, *g, gc2_x + j, gc2_y + k );

        }
    }

    color_avg = ( ( ( red / 9 ) & 0xff ) << 16 )
                ||
                ( ( ( green / 9 ) & 0xff ) << 8 )
                ||
                ( ( blue / 9 ) & 0xff );

    return ( color_avg );

}

