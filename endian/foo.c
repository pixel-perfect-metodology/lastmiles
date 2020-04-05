

#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdint.h>

int main(int argc,char *argv[])
{
    
    uint32_t a=1024;
    uint8_t bytes[4];
    uint8_t *p=(uint8_t*)&a;
    bytes[0]=p[0];
    bytes[1]=p[1];
    bytes[2]=p[2];
    bytes[3]=p[3];

    uint8_t bmp_header[12] = { 0, 1, 2, 3,
                               0, 4, 0, 0,
                               0,12,34, 1 };
    
    printf("%u\n",(bmp_header[7]<<24)|(bmp_header[6]<<16)|(bmp_header[5]<<8)|bmp_header[4]);
    return 0;
}

