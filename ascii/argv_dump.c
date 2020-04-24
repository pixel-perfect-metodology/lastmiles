
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <sys/utsname.h>

int sysinfo(void);
uint64_t system_memory(void);

int main(int argc, char *argv[]) 
{

    int foo = 0;
    int char_count_total = 0;
    setlocale( LC_ALL, "C" );
    sysinfo();

    /* The C standard really doesn't clearly say how a
     * pointer will be printed. C11 ISO/IEC 9899:2011
     * the idea is described in section 7.21.6.1 para8.
     * Maybe you get a leading 0x and maybe you don't and
     * really we are printing a void* type pointer.
     * Just use C99 <inttypes.h> and uintptr_t instead. */

    printf ( "argc = %i\n", argc );
    printf ( "&argc = %p\n", &argc );
    printf ( "argv = %p\n", argv );

    for ( foo = 0; foo < argc; foo++ ) {
        printf ( "%2i chars for argv[%2i] = \"%s\"\n", strlen(argv[foo]), foo, argv[foo] );
        char_count_total += strlen(argv[foo]);
    }

    for ( foo = 0; foo < ( char_count_total + argc ); foo++ ) {
        printf("%02x ", ((uint8_t*)argv[0])[foo] );
    }

    printf("\n");

    return ( 42 );

}

