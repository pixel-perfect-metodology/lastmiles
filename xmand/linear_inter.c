
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
#include <stdint.h>
#include <unistd.h>

uint32_t linear_inter( uint8_t  in_val,
                       uint32_t low_col, uint32_t high_col,
                       uint8_t  low_val, uint8_t upper_val)
{
    /* in_val is some number that should fall between
     *        the low_val and upper_val. If not then
     *        just assume low_val or upper_val as
     *        needed.
     *
     * low_col is the actual RGB value at the low end
     *         of the scale
     *
     * high_col is the RGB colour at the high end of the
     *           scale
     *
     * low_val and upper_val are the possible range values
     *          for the in_val
     *
     * How to do a linear interpolation between two 32-bit colour
     * values?  We need a smooth function :
     *
     *    uint32_t cpixel = ( uint8_t   red_val << 16 )
     *                       + ( uint8_t green_val << 8 )
     *                       +   uint8_t  blue_val
     **/

    uint8_t red, green, blue;
    uint8_t lower_red, upper_red;
    uint8_t lower_green, upper_green;
    uint8_t lower_blue, upper_blue;
    uint32_t cpixel;

    if (    ( high_col & (uint32_t)0xff0000 )
         <= (  low_col & (uint32_t)0xff0000 ) ) {

        lower_red = (uint8_t)( ( high_col & (uint32_t)0xff0000 ) >> 16 );
        upper_red = (uint8_t)( (  low_col & (uint32_t)0xff0000 ) >> 16 );

    } else {

        upper_red = (uint8_t)( ( high_col & (uint32_t)0xff0000 ) >> 16 );
        lower_red = (uint8_t)( (  low_col & (uint32_t)0xff0000 ) >> 16 );

    }

    if (    ( high_col & (uint32_t)0x00ff00 )
         <= (  low_col & (uint32_t)0x00ff00 ) ) {

        lower_green = (uint8_t)( ( high_col & (uint32_t)0x00ff00 ) >> 8 );
        upper_green = (uint8_t)( (  low_col & (uint32_t)0x00ff00 ) >> 8 );

    } else {

        upper_green = (uint8_t)( ( high_col & (uint32_t)0x00ff00 ) >> 8 );
        lower_green = (uint8_t)( (  low_col & (uint32_t)0x00ff00 ) >> 8 );

    }

    if (    ( high_col & (uint32_t)0x0000ff )
         <= (  low_col & (uint32_t)0x0000ff ) ) {

        lower_blue = (uint8_t)( high_col & (uint32_t)0x0000ff );
        upper_blue = (uint8_t)(  low_col & (uint32_t)0x0000ff );

    } else {

        upper_blue = (uint8_t)( high_col & (uint32_t)0x0000ff );
        lower_blue = (uint8_t)(  low_col & (uint32_t)0x0000ff );

    }

    red = lower_red
           + ( upper_red - lower_red )
             * ( in_val - low_val ) / ( upper_val - low_val );

    green = lower_green
           + ( upper_green - lower_green )
             * ( in_val - low_val ) / ( upper_val - low_val );

    blue = lower_blue
           + ( upper_blue - lower_blue )
             * ( in_val - low_val ) / ( upper_val - low_val );

    cpixel = ( (uint32_t)red << 16 )
           | ( (uint32_t)green << 8 )
           |   (uint32_t)blue;

    return ( cpixel );

}

