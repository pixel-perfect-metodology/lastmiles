
void dump_atime(char *foopath){
    char atime[64]; /* guess */
    struct stat statbuf;
    if (stat(foopath,&statbuf) == (-1)){
        perror("no such file?");
        exit (EXIT_FAILURE);
    }else{
        strcpy(atime, ctime(&statbuf.st_atime));
        /* bail out anyways */
    }
}

