
long double timespec_to_ld( struct timespec t ) {

    char buffer[64] = "";
    int bytes_formatted;
    long double ld_t = ( long double ) 0.0 ;

    bytes_formatted = sprintf ( buffer, "%10lu.%-9lu",
                                t.tv_sec, t.tv_nsec );

    if ( bytes_formatted > 0 )
        ld_t = strtold( buffer, NULL );

    if ( ld_t > ( ( long double ) 0.0 ) )
        return  ld_t;
    else
        return ( long double ) -1.0;

}

