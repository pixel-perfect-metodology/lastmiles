
#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <locale.h>

int sysinfo(void);

int main ( int argc, char **argv ) 
{

    /* we shall define the value we need as the 
     * little endian bytes for a 32-bit interger
     * at bmp_header[4] to bmp_header[7] */

    uint8_t bmp_header[12] = { 0, 1, 2, 3,
                               0, 4, 0, 0,
                               0,12,34, 1 };

    /* thus the bytes 0x00 0x04 0x00 0x00 must be 
     * a little endian integer of value 1024 decimal */

    uint32_t pi =  31415926; /* should be 0x1DF5E76 */ 
    printf("DBUG : our test pi = 0x%08X\n", pi );
    printf("     : could be      0x01DF5E76\n");

    /* this is the bmp_width data that we are really trying
     * to extract from that array bmp_header. We shall set
     * this to a reasonable interpretation of the bytes
     * once we know what this machine "thinks" and integer
     * really is. */
    int bmp_width;

    /* this is a test integer data value where we want to
     * know where is the one bit that is turned "on". Is
     * it a four byte integer where the memory for a value
     * of 1 looks like   0x01000000h or is it the more 
     * normal looking    0x00000001h which is big endian
     * byte order? */
    int j = 1;

    /* lets stick to a trivial C or POSIX locale */
    setlocale( LC_ALL, "C" );

    /* output some basic information about this machine */
    sysinfo();

    int little_endian = (*(uint8_t*)&j == 1) ? 1 : 0;

    /* We need to be aware that we can violate the C standards 
     * in that section 6.3.2.3 number 6 states : 
     *     "A pointer to an object type may be converted
     *      to a pointer to a different object type. If
     *      the resulting pointer is not correctly aligned
     *      for the referenced type, the behavior is
     *      undefined."
     *
     * In the case of little endian machine we could do 
     *
     *     foo = ( data[0]<<0) 
     *             | (data[1]<<8) 
     *             | (data[2]<<16)
     *             | (data[3]<<24) );
     *
     * with the big endian approach being similar 
     *
     *     foo = ( data[3]<<0) 
     *             | (data[2]<<8) 
     *             | (data[1]<<16)
     *             | (data[0]<<24) );
     *
     * where "data" elements must be simple 8-bit byte objects
     * such as uint8_t primitive types.
     *
     *
     * The casting of pointers from one type to another type
     * may be dangerous and thus this should be avoided :
     *
     *    bmp_width = *( (uint32_t *) &bar );
     *
     */

    if ( little_endian ) {

        printf ("This is a little endian machine.\n");

        printf ("DBUG : (uint32_t)bmp_header[7]           = %08X\n",  (uint32_t)bmp_header[7] );
        printf ("DBUG : ( ((uint32_t)bmp_header[6])<<8  ) = %08X\n",  ( ((uint32_t)bmp_header[6])<<8  ) );
        printf ("DBUG : ( ((uint32_t)bmp_header[5])<<16 ) = %08X\n",  ( ((uint32_t)bmp_header[5])<<16 ) );
        printf ("DBUG : ( ((uint32_t)bmp_header[4])<<24 ) = %08X\n",  ( ((uint32_t)bmp_header[4])<<24 ) );

        bmp_width =  ( ( (uint32_t)bmp_header[7] )     ) 
                   | ( ( (uint32_t)bmp_header[6] )<<8  ) 
                   | ( ( (uint32_t)bmp_header[5] )<<16 )
                   | ( ( (uint32_t)bmp_header[4] )<<24 );

    } else {

        printf ("This is a big endian machine.\n");

        bmp_width =  ( bmp_header[4]     ) 
                   | ( bmp_header[5]<<8  ) 
                   | ( bmp_header[6]<<16 )
                   | ( bmp_header[7]<<24 );

    }

    printf ( "bmp_width = %i\n", bmp_width );

    /* should be 1024 */

    return EXIT_SUCCESS;

}

