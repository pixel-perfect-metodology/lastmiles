# 1 "argv_dump.c"

 
# 14

# 1 "/usr/include/stdio.h"
 
# 4

 
 

 
 
 

 
# 15

# 18

#ident	"@(#)stdio.h	1.86	13/09/11 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 7

#ident	"@(#)feature_tests.h	1.26	11/04/12 SMI"

# 1 "/usr/include/sys/ccompile.h"
 
# 5

# 8

#ident	"@(#)ccompile.h	1.2	04/11/08 SMI"

 
# 15

# 19

 
# 29

# 84

# 86

# 88

 
# 92

# 100


# 105

# 1 "/usr/include/sys/isa_defs.h"
 
# 4

# 7

#ident	"@(#)isa_defs.h	1.30	11/03/31 SMI"

 
# 191

# 195

 
# 395

 
# 402

 
# 412

 
# 433

 
# 442

 
# 473

 
# 482

# 484

 
# 495

# 499

 
# 506

# 510

# 514

# 12 "/usr/include/sys/feature_tests.h"

# 16

 
# 30

 
# 61

# 65

 
# 110

# 117

 
# 121

# 125

 
# 165

 
# 193

 
# 244

 
# 278

 
# 302

 
# 307

 
# 328

 
# 344

 
# 357

 
# 363

 
# 369

 
# 375

# 379

# 22 "/usr/include/stdio.h"

# 26

 
# 33

 
# 46

 
# 59

# 61

# 65

# 1 "/usr/include/iso/stdio_iso.h"
 
# 5

 
 

 
 
 

 
# 24

 
# 28

# 31

#ident	"@(#)stdio_iso.h	1.8	05/08/16 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 1 "/usr/include/sys/va_list.h"
 
 

 
 
 

 
# 12

# 15

#ident	"@(#)va_list.h	1.15	04/11/19 SMI"

 
# 37

 
# 50

# 1 "/usr/include/sys/isa_defs.h"
 
# 4

# 52 "/usr/include/sys/va_list.h"

# 56

# 62

# 64

# 67

# 74

# 108

typedef  void *__va_list;

# 112

# 116

# 1 "/usr/include/stdio_tag.h"
 
# 5

# 8

#ident	"@(#)stdio_tag.h	1.4	04/09/28 SMI"

# 14

# 21
typedef struct  __FILE __FILE;
# 23

# 27

# 1 "/usr/include/stdio_impl.h"
 
# 5

# 8

#ident	"@(#)stdio_impl.h	1.15	07/03/05 SMI"

# 1 "/usr/include/sys/isa_defs.h"
 
# 4

# 12 "/usr/include/stdio_impl.h"

# 16

# 18

# 20

struct  __FILE {
	long	__pad[16];
};

# 26

# 51

# 55

# 38 "/usr/include/iso/stdio_iso.h"

 
# 46

# 50

# 67

# 71

# 74
typedef	__FILE FILE;
# 76

# 80
typedef unsigned long	size_t;		 
# 85

# 87
typedef long		fpos_t;
# 91

# 95

# 103

# 105

 
# 117

# 119

# 125

# 130

# 134

# 137

# 142

# 144

# 146
extern __FILE	__iob[ 20 ];
# 156

# 160

# 186

# 188

extern int	remove(const char *);
extern int	rename(const char *, const char *);
extern FILE	*tmpfile(void);
extern char	*tmpnam(char *);
extern int	fclose(FILE *);
extern int	fflush(FILE *);
extern FILE	*fopen(const char * restrict , const char * restrict );
extern FILE	*freopen(const char * restrict ,
			const char * restrict , FILE * restrict );
extern void	setbuf(FILE * restrict , char * restrict );
extern int	setvbuf(FILE * restrict , char * restrict , int,
			size_t);
 
extern int	fprintf(FILE * restrict , const char * restrict , ...);
 
extern int	fscanf(FILE * restrict , const char * restrict , ...);
 
extern int	printf(const char * restrict , ...);
 
extern int	scanf(const char * restrict , ...);
 
extern int	sprintf(char * restrict , const char * restrict , ...);
 
extern int	sscanf(const char * restrict ,
			const char * restrict , ...);
extern int	vfprintf(FILE * restrict , const char * restrict ,
			__va_list);
extern int	vprintf(const char * restrict , __va_list);
extern int	vsprintf(char * restrict , const char * restrict ,
			__va_list);
extern int	fgetc(FILE *);
extern char	*fgets(char * restrict , int, FILE * restrict );
extern int	fputc(int, FILE *);
extern int	fputs(const char * restrict , FILE * restrict );
# 225
extern int	getc(FILE *);
extern int	putc(int, FILE *);
# 230
extern int	getchar(void);
extern int	putchar(int);
# 233
extern char	*gets(char *);
extern int	puts(const char *);
extern int	ungetc(int, FILE *);
extern size_t	fread(void * restrict , size_t, size_t,
	FILE * restrict );
extern size_t	fwrite(const void * restrict , size_t, size_t,
	FILE * restrict );
# 241
extern int	fgetpos(FILE * restrict , fpos_t * restrict );
extern int	fsetpos(FILE *, const fpos_t *);
# 244
extern int	fseek(FILE *, long, int);
extern long	ftell(FILE *);
extern void	rewind(FILE *);
# 249
extern void	clearerr(FILE *);
extern int	feof(FILE *);
extern int	ferror(FILE *);
# 253
extern void	perror(const char *);

# 259

# 310

# 314

# 316

# 340

# 342

# 352

# 366

# 368

# 370

# 374

# 67 "/usr/include/stdio.h"

 
# 75

 
# 130

 
# 1 "/usr/include/iso/stdio_c99.h"
 
# 5

 
# 16

# 19

#ident	"@(#)stdio_c99.h	1.2	04/03/29 SMI"

# 25

 
# 33

# 36

# 52

# 54
extern int vfscanf(FILE * restrict , const char * restrict , __va_list);
extern int vscanf(const char * restrict , __va_list);
extern int vsscanf(const char * restrict , const char * restrict ,
		__va_list);
# 68
extern int snprintf(char * restrict , size_t, const char * restrict ,
	...);
extern int vsnprintf(char * restrict , size_t, const char * restrict ,
	__va_list);
# 76

# 78

# 82

# 136 "/usr/include/stdio.h"

# 140

# 145
typedef long		off_t;
# 158

# 166

 
# 174
typedef	__va_list va_list;
# 176

# 179

# 181

 
# 186

# 188

# 192

# 195

# 197
extern unsigned char	 _sibuf[], _sobuf[];
# 199

 
# 227

# 232

 
# 247

# 249

# 255

# 265

 
# 271

extern FILE	*fdopen(int, const char *);
extern char	*ctermid(char *);
extern int	fileno(FILE *);

# 277

 
# 284
extern void	flockfile(FILE *);
extern int	ftrylockfile(FILE *);
extern void	funlockfile(FILE *);
extern int	getc_unlocked(FILE *);
extern int	getchar_unlocked(void);
extern int	putc_unlocked(int, FILE *);
extern int	putchar_unlocked(int);

# 293

 
# 299
extern FILE	*popen(const char *, const char *);
extern char	*tempnam(const char *, const char *);
extern int	pclose(FILE *);
# 305

 
# 315

# 317

 
# 322
extern int	fseeko(FILE *, off_t, int);
extern off_t	ftello(FILE *);
# 325

 
# 340

# 416

# 418

# 443

# 445

# 449

# 1 "/usr/include/stdlib.h"
 
# 4

 
 

 
 
 

# 14

#ident	"@(#)stdlib.h	1.52	12/08/01 SMI"

# 1 "/usr/include/iso/stdlib_iso.h"
 
# 5

 
 

 
 
 

 
# 24

# 27

#ident	"@(#)stdlib_iso.h	1.9	04/09/28 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 31 "/usr/include/iso/stdlib_iso.h"

# 35

# 37
extern unsigned char	__ctype[];
# 43

# 47

typedef	struct {
	int	quot;
	int	rem;
} div_t;

typedef struct {
	long	quot;
	long	rem;
} ldiv_t;

# 66

# 74

# 78

 
# 97
typedef	int	wchar_t;
# 104

# 106

extern void abort(void) ;
extern int abs(int);
extern int atexit(void (*)(void));
extern double atof(const char *);
extern int atoi(const char *);
extern long int atol(const char *);
extern void *bsearch(const void *, const void *, size_t, size_t,
	int (*)(const void *, const void *));
# 121
extern void *calloc(size_t, size_t);
extern div_t div(int, int);
extern void exit(int)
	;
extern void free(void *);
extern char *getenv(const char *);
extern long int labs(long);
extern ldiv_t ldiv(long, long);
extern void *malloc(size_t);
extern int mblen(const char *, size_t);
extern size_t mbstowcs(wchar_t * restrict , const char * restrict ,
	size_t);
extern int mbtowc(wchar_t * restrict , const char * restrict , size_t);
extern void qsort(void *, size_t, size_t, int (*)(const void *, const void *));
# 140
extern int rand(void);
extern void *realloc(void *, size_t);
extern void srand(unsigned int);
extern double strtod(const char * restrict , char ** restrict );
extern long int strtol(const char * restrict , char ** restrict , int);
extern unsigned long int strtoul(const char * restrict ,
	char ** restrict , int);
extern int system(const char *);
extern int wctomb(char *, wchar_t);
extern size_t wcstombs(char * restrict , const wchar_t * restrict ,
	size_t);

# 158

# 191

# 195

# 199

# 1 "/usr/include/iso/stdlib_c99.h"
 
# 5

 
# 16

# 19

#ident	"@(#)stdlib_c99.h	1.2	04/03/29 SMI"

# 25

 
# 33

# 35
typedef struct {
	long long	quot;
	long long	rem;
} lldiv_t;
# 40

# 42

# 45

extern void _Exit(int);
extern float strtof(const char * restrict , char ** restrict );
extern long double strtold(const char * restrict , char ** restrict );

# 51
extern long long atoll(const char *);
extern long long llabs(long long);
extern lldiv_t lldiv(long long, long long);
extern long long strtoll(const char * restrict , char ** restrict ,
	int);
extern unsigned long long strtoull(const char * restrict ,
	char ** restrict , int);
# 59

# 61

# 81

# 85

# 19 "/usr/include/stdlib.h"

# 1 "/usr/include/sys/wait.h"
 
 

 
 
 

 
# 12

# 15

#ident	"@(#)wait.h	1.23	04/06/03 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 19 "/usr/include/sys/wait.h"

# 1 "/usr/include/sys/types.h"
 
 

 
 
 

 
# 11

# 14

#ident	"@(#)types.h	1.90	14/04/07 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 1 "/usr/include/sys/isa_defs.h"
 
# 4

# 19 "/usr/include/sys/types.h"

 
# 1 "/usr/include/sys/machtypes.h"
 
# 5

 
 

 
 
 

# 15

#ident	"@(#)machtypes.h	1.13	99/05/04 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 19 "/usr/include/sys/machtypes.h"

# 23

 
# 29

# 36

typedef	unsigned char	lock_t;		 

# 42

# 24 "/usr/include/sys/types.h"

 
# 1 "/usr/include/sys/int_types.h"
 
# 5

# 8

#ident	"@(#)int_types.h	1.10	04/09/28 SMI"

 
# 33

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 35 "/usr/include/sys/int_types.h"

# 39

 
# 54
typedef char			int8_t;
# 60
typedef short			int16_t;
typedef int			int32_t;
# 64
typedef long			int64_t;
# 71

typedef unsigned char		uint8_t;
typedef unsigned short		uint16_t;
typedef unsigned int		uint32_t;
# 76
typedef unsigned long		uint64_t;
# 82

 
# 88
typedef int64_t			intmax_t;
typedef uint64_t		uintmax_t;
# 94

 
# 101
typedef long			intptr_t;
typedef unsigned long		uintptr_t;
# 107

 
# 113
typedef char			int_fast8_t;
# 119
typedef int			int_fast16_t;
typedef int			int_fast32_t;
# 122
typedef long			int_fast64_t;
# 128

typedef unsigned char		uint_fast8_t;
typedef unsigned int		uint_fast16_t;
typedef unsigned int		uint_fast32_t;
# 133
typedef unsigned long		uint_fast64_t;
# 139

 
# 145
typedef char			int_least8_t;
# 151
typedef short			int_least16_t;
typedef int			int_least32_t;
# 154
typedef long			int_least64_t;
# 160

typedef unsigned char		uint_least8_t;
typedef unsigned short		uint_least16_t;
typedef unsigned int		uint_least32_t;
# 165
typedef unsigned long		uint_least64_t;
# 171

# 175

# 37 "/usr/include/sys/types.h"

# 41

# 45

 
# 52
typedef	long long		longlong_t;
typedef	unsigned long long	u_longlong_t;
# 65

 
# 72
typedef int32_t		t_scalar_t;
typedef uint32_t	t_uscalar_t;
# 78

 
# 82
typedef	unsigned char	uchar_t;
typedef	unsigned short	ushort_t;
typedef	unsigned int	uint_t;
typedef	unsigned long	ulong_t;

typedef	char		*caddr_t;	 
typedef	long		daddr_t;	 
typedef	short		cnt_t;		 

# 94
typedef	long	ptrdiff_t;		 
# 99

 
# 103
typedef	ulong_t		pfn_t;		 
typedef	ulong_t		pgcnt_t;	 
typedef	long		spgcnt_t;	 

typedef	uchar_t		use_t;		 
typedef	short		sysid_t;
typedef	short		index_t;
typedef void		*timeout_id_t;	 
typedef void		*bufcall_id_t;	 

 
# 139

# 141
typedef ulong_t		ino_t;		 
typedef long		blkcnt_t;	 
typedef ulong_t		fsblkcnt_t;	 
typedef ulong_t		fsfilcnt_t;	 
# 151

# 165

# 167
typedef	int		blksize_t;	 
# 171

# 173
typedef enum { _B_FALSE, _B_TRUE } boolean_t;
# 177

 
# 191
typedef int64_t		pad64_t;
typedef	uint64_t	upad64_t;
# 204

typedef union {
	long double	_q;
	int32_t		_l[4];
} pad128_t;

typedef union {
	long double	_q;
	uint32_t	_l[4];
} upad128_t;

typedef	longlong_t	offset_t;
typedef	u_longlong_t	u_offset_t;
typedef u_longlong_t	len_t;
typedef	u_longlong_t	diskaddr_t;
# 222

 
# 237

# 239
typedef union {
	offset_t	_f;	 
	struct {
		int32_t	_u;	 
		int32_t	_l;	 
	} _p;
} lloff_t;
# 247

# 257

# 259
typedef union {
	longlong_t	_f;	 
	struct {
		int32_t	_u;	 
		int32_t	_l;	 
	} _p;
} lldaddr_t;
# 267

typedef uint_t k_fltset_t;	 

 
# 280
typedef int		id_t;
# 284

typedef id_t		lgrp_id_t;	 

 
# 291
typedef uint_t		useconds_t;	 

# 295
typedef long	suseconds_t;	 
# 297

 
# 302
typedef uint_t	major_t;	 
typedef uint_t	minor_t;	 
# 308

 
# 312
typedef short	pri_t;

 
# 318
typedef ushort_t	cpu_flag_t;

 
# 331
typedef	ushort_t o_mode_t;		 
typedef short	o_dev_t;		 
typedef	ushort_t o_uid_t;		 
typedef	o_uid_t	o_gid_t;		 
typedef	short	o_nlink_t;		 
typedef short	o_pid_t;		 
typedef ushort_t o_ino_t;		 


 
# 343
typedef	int	key_t;			 
# 345
typedef	uint_t	mode_t;			 
# 349

# 353
typedef	int	uid_t;			 
# 358

typedef	uid_t	gid_t;			 

typedef id_t    taskid_t;
typedef id_t    projid_t;
typedef	id_t	poolid_t;
typedef id_t	zoneid_t;
typedef id_t	ctid_t;

 
# 372
typedef	uint_t	pthread_t;	 
typedef	uint_t	pthread_key_t;	 

 
# 380

typedef	struct _pthread_mutex {		 
	struct {
		uint16_t	__pthread_mutex_flag1;
		uint8_t		__pthread_mutex_flag2;
		uint8_t		__pthread_mutex_ceiling;
		uint16_t 	__pthread_mutex_type;
		uint16_t 	__pthread_mutex_magic;
	} __pthread_mutex_flags;
	union {
		struct {
			uint8_t	__pthread_mutex_pad[8];
		} __pthread_mutex_lock64;
		struct {
			uint32_t __pthread_ownerpid;
			uint32_t __pthread_lockword;
		} __pthread_mutex_lock32;
		upad64_t __pthread_mutex_owner64;
	} __pthread_mutex_lock;
	upad64_t __pthread_mutex_data;
} pthread_mutex_t;

typedef	struct _pthread_cond {		 
	struct {
		uint8_t		__pthread_cond_flag[4];
		uint16_t 	__pthread_cond_type;
		uint16_t 	__pthread_cond_magic;
	} __pthread_cond_flags;
	upad64_t __pthread_cond_data;
} pthread_cond_t;

 
# 414
typedef	struct _pthread_rwlock {	 
	int32_t		__pthread_rwlock_readers;
	uint16_t	__pthread_rwlock_type;
	uint16_t	__pthread_rwlock_magic;
	pthread_mutex_t	__pthread_rwlock_mutex;
	pthread_cond_t	__pthread_rwlock_readercv;
	pthread_cond_t	__pthread_rwlock_writercv;
} pthread_rwlock_t;

 
# 426
typedef struct {
	uint32_t	__pthread_barrier_count;
	uint32_t	__pthread_barrier_current;
	upad64_t	__pthread_barrier_cycle;
	upad64_t	__pthread_barrier_reserved;
	pthread_mutex_t	__pthread_barrier_lock;
	pthread_cond_t	__pthread_barrier_cond;
} pthread_barrier_t;

typedef	pthread_mutex_t	pthread_spinlock_t;

 
# 440
typedef struct _pthread_attr {
	void	*__pthread_attrp;
} pthread_attr_t;

 
# 447
typedef struct _pthread_mutexattr {
	void	*__pthread_mutexattrp;
} pthread_mutexattr_t;

 
# 454
typedef struct _pthread_condattr {
	void	*__pthread_condattrp;
} pthread_condattr_t;

 
# 461
typedef	struct _once {
	upad64_t	__pthread_once_pad[4];
} pthread_once_t;

 
# 469
typedef struct _pthread_rwlockattr {
	void	*__pthread_rwlockattrp;
} pthread_rwlockattr_t;

 
# 477
typedef struct {
	void	*__pthread_barrierattrp;
} pthread_barrierattr_t;

typedef ulong_t	dev_t;			 

# 484
typedef	uint_t nlink_t;			 
typedef int	pid_t;			 
# 490

# 499

# 503
typedef long	ssize_t;	 
# 508

# 511
typedef	long		time_t;	 
# 513

# 516
typedef	long		clock_t;  
# 518

# 521
typedef	int	clockid_t;	 
# 523

# 526
typedef	int	timer_t;	 
# 528

# 632

 
# 642
 
# 644

# 648

# 21 "/usr/include/sys/wait.h"

# 1 "/usr/include/sys/resource.h"
 
# 5

 
 

 
 
 

# 15

#ident	"@(#)resource.h	1.37	07/02/07 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 19 "/usr/include/sys/resource.h"

# 1 "/usr/include/sys/types.h"
 
 

 
 
 

 
# 11

# 1 "/usr/include/sys/time.h"
 
 

 
 
 

 
# 13

 
# 17

# 20

#ident	"@(#)time.h	2.79	13/07/01 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 24 "/usr/include/sys/time.h"

 
# 29

# 33

# 37

# 42

# 47

struct timeval {
	time_t		tv_sec;		 
	suseconds_t	tv_usec;	 
};

# 74

# 77

# 87

# 91

 
# 1 "/usr/include/sys/types.h"
 
 

 
 
 

 
# 11

# 100 "/usr/include/sys/time.h"

# 104

# 133

# 135
 
# 142
				 
				 
# 145
				 
				 

# 149
struct	itimerval {
	struct	timeval it_interval;	 
	struct	timeval it_value;	 
};

# 178


# 190

# 192

 
# 196
typedef	longlong_t	hrtime_t;

# 321

# 329

# 338

# 341

# 343

int getitimer(int, struct itimerval *);
int utimes(const char *, const struct timeval *);
# 347
int setitimer(int, const struct itimerval * restrict ,
	struct itimerval * restrict );
# 353

# 361

 
# 393

# 396

# 401
int gettimeofday(struct timeval * restrict , void * restrict );
# 406

# 408

 
# 423

 
# 1 "/usr/include/sys/select.h"
 
# 5

 

 
 
 

# 14

#ident	"@(#)select.h	1.19	04/01/28 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 18 "/usr/include/sys/select.h"

# 1 "/usr/include/sys/time_impl.h"
 
# 5

 
# 15

# 18

#ident	"@(#)time_impl.h	1.11	05/05/19 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 22 "/usr/include/sys/time_impl.h"

# 26

# 28

# 33

 
# 37

typedef struct  timespec {		 
	time_t		tv_sec;		 
	long		tv_nsec;	 
} timespec_t;

# 61

typedef struct timespec timestruc_t;	 

 
# 68

# 72

 
# 76
typedef struct itimerspec {		 
	struct timespec	it_interval;	 
	struct timespec	it_value;	 
} itimerspec_t;

# 98

# 100

# 109

# 113

# 116

# 120

# 1 "/usr/include/sys/time.h"
 
 

 
 
 

 
# 13

 
# 17

# 25 "/usr/include/sys/select.h"

# 29


# 32
 
# 44
typedef struct {		 
	unsigned int	__sigbits[4];
} sigset_t;

# 58

# 60

 
# 86

# 90
typedef	long	fds_mask;

 
# 102

# 107

# 114

# 118
typedef	struct __fd_set {
# 120
	long	fds_bits[ ( ( ( 65536 ) + ( ( ( sizeof ( fds_mask ) * 8 ) ) - 1 ) ) / ( ( sizeof ( fds_mask ) * 8 ) ) )];
} fd_set;

# 125

# 128

# 131

# 137

# 140
extern int select(int, fd_set * restrict , fd_set * restrict ,
	fd_set * restrict , struct timeval * restrict );

# 144
extern int pselect(int, fd_set * restrict , fd_set * restrict ,
	fd_set * restrict , const struct timespec * restrict ,
	const sigset_t * restrict );
# 148

# 156

# 160

# 435 "/usr/include/sys/time.h"

# 437

# 441

# 22 "/usr/include/sys/resource.h"

# 26

 
# 40

 
# 52

# 54

# 56

typedef	unsigned long	rlim_t;

# 62

# 88

# 105

struct rlimit {
	rlim_t	rlim_cur;		 
	rlim_t	rlim_max;		 
};

 
# 126

 
# 133

# 136

# 143


struct	rusage {
	struct timeval ru_utime;	 
	struct timeval ru_stime;	 
	long	ru_maxrss;		 
	long	ru_ixrss;		 
	long	ru_idrss;		 
	long	ru_isrss;		 
	long	ru_minflt;		 
	long	ru_majflt;		 
	long	ru_nswap;		 
	long	ru_inblock;		 
	long	ru_oublock;		 
	long	ru_msgsnd;		 
	long	ru_msgrcv;		 
	long	ru_nsignals;		 
	long	ru_nvcsw;		 
	long	ru_nivcsw;		 
};

# 168

# 191


# 200

# 204


# 219

# 234

# 236

extern int setrlimit(int, const struct rlimit *);
extern int getrlimit(int, struct rlimit *);

 
# 246

extern int getpriority(int, id_t);
extern int setpriority(int, id_t, int);
extern int getrusage(int, struct rusage *);

# 268

# 270

# 274

# 1 "/usr/include/sys/siginfo.h"
 
# 5

 
 

 
# 13

 
 
 

# 20

#ident	"@(#)siginfo.h	1.59	04/07/15 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 1 "/usr/include/sys/types.h"
 
 

 
 
 

 
# 11

# 25 "/usr/include/sys/siginfo.h"

# 29

# 32

 
# 39
union sigval {
	int	sival_int;	 
	void	*sival_ptr;	 
};
# 44

# 55

# 64

# 67

 
# 74
struct sigevent {
	int		sigev_notify;	 
	int		sigev_signo;	 
	union sigval	sigev_value;	 
	void		(*sigev_notify_function)(union sigval);
	pthread_attr_t	*sigev_notify_attributes;
	int		__sigev_pad2;
};
# 83

 
# 89

# 104

# 106

# 109
 
# 113

# 116

# 127

# 129
 
# 133

# 1 "/usr/include/sys/machsig.h"
 
 

 
 
 

 
# 12

# 15

#ident	"@(#)machsig.h	1.15	99/08/15 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 19 "/usr/include/sys/machsig.h"

# 23

 
# 27

 
# 35

# 37

 
# 41

# 50

# 54

 
# 58

# 64

 
# 68

# 80

 
# 84

# 90

 
# 94

# 103

# 105

# 109

# 135 "/usr/include/sys/siginfo.h"

 
# 139

# 149

 
# 153

# 160

# 164

 
# 168

# 175

# 179

# 181

# 191

# 194

# 202

 
# 1 "/usr/include/sys/time_std_impl.h"
 
# 5

 
# 15

# 18

#ident	"@(#)time_std_impl.h	1.3	04/01/21 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 22 "/usr/include/sys/time_std_impl.h"

# 26

# 31

typedef	struct	_timespec {
	time_t	__tv_sec;	 
	long	__tv_nsec;	 
} _timespec_t;

typedef	struct	_timespec	_timestruc_t;	 

# 42

# 217 "/usr/include/sys/siginfo.h"

 
# 1 "/usr/include/sys/types.h"
 
 

 
 
 

 
# 11

# 225 "/usr/include/sys/siginfo.h"

# 229
typedef struct {
# 231
	int	si_signo;			 
	int 	si_code;			 
	int	si_errno;			 
# 235
	int	si_pad;		 
# 237
	union {

		int	__pad[ ( ( 256 / sizeof ( int ) ) - 4 )];		 

		struct {			 
			pid_t	__pid;		 
			union {
				struct {
					uid_t	__uid;
# 248
					union sigval	__value;
# 252
				} __kill;
				struct {
					clock_t __utime;
					int	__status;
					clock_t __stime;
				} __cld;
			} __pdata;
			ctid_t	__ctid;		 
			zoneid_t __zoneid;	 
		} __proc;

		struct {	 
			void 	*__addr;	 
			int	__trapno;	 
			caddr_t	__pc;		 
		} __fault;

		struct {			 
		 
			int	__fd;		 
			long	__band;
		} __file;

		struct {			 
			caddr_t	__faddr;	 
# 280
			_timestruc_t __tstamp;	 
# 282
			short	__syscall;	 
			char	__nsysarg;	 
			char	__fault;	 
			long	__sysarg[8];	 
			int	__mstate[10];	 
		} __prof;

		struct {			 
			int32_t	__entity;	 
		} __rctl;
	} __data;

} siginfo_t;

# 361

 
# 366

# 443

# 466

# 468


# 476

# 480

# 1 "/usr/include/sys/procset.h"
 
# 4

 
 

 
 
 

# 14

#ident	"@(#)procset.h	1.26	11/04/18 SMI"

# 20

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 1 "/usr/include/sys/types.h"
 
 

 
 
 

 
# 11

# 1 "/usr/include/sys/signal.h"
 
# 4

 
 

 
 
 

# 14

#ident	"@(#)signal.h	1.67	13/09/11 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 1 "/usr/include/sys/iso/signal_iso.h"
 
# 5

 
 

 
 
 

 
# 24

# 27

#ident	"@(#)signal_iso.h	1.6	03/05/02 SMI"

# 1 "/usr/include/sys/unistd.h"
 
# 9

 
# 13

 
# 19

# 22

#ident	"@(#)unistd.h	1.46	12/01/17 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 26 "/usr/include/sys/unistd.h"

# 30

 

# 34

 
# 38
 
# 43
 
# 48

 
# 66

 
# 85

 

 
# 97
 
# 102
 
# 107
 
# 135
 
# 156

 
# 160

 
# 166

 
# 171

# 174

 
# 190

# 196

 
# 217

 
# 226

 
# 267

 

 
# 280
 
# 284
 
# 297
 
# 305

 
# 311

 
# 318

# 326

# 334

 
# 347

# 351

# 355

# 360

# 376

 
# 381

# 385

# 31 "/usr/include/sys/iso/signal_iso.h"

# 35

# 79

 
# 83
extern long _sysconf(int);	 
# 86

# 106

# 111

# 113

# 117

# 124

# 128

# 19 "/usr/include/sys/signal.h"

# 23

# 26

# 30
 
# 1 "/usr/include/sys/siginfo.h"
 
# 5

 
 

 
# 13

 
 
 

# 35 "/usr/include/sys/signal.h"

 
# 54

typedef	struct {
	unsigned int	__sigbits[2];
} k_sigset_t;

 
# 69

 
# 73
struct sigaction {
	int sa_flags;
	union {
# 79
		void (*_handler)();
# 84
		void (*_sigaction)(int, siginfo_t *, void *);
# 86
	}	_funcptr;
	sigset_t sa_mask;
# 91
};
# 94

# 110

 
# 113

# 115

# 119

			 

 
# 127

# 133

# 138

 
# 141

# 150

# 153

# 156

 
# 163
typedef struct {
# 165
	void	*ss_sp;
	size_t	ss_size;
	int	ss_flags;
} stack_t;

# 181

# 183

# 185

# 215

 
# 220
struct sigstack {
	void	*ss_sp;
	int	ss_onstack;
};
# 225

 
# 1 "/usr/include/sys/ucontext.h"
 
 

 
 
 

 
# 12

# 15

#ident	"@(#)ucontext.h	1.31	03/12/18 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 19 "/usr/include/sys/ucontext.h"

# 1 "/usr/include/sys/types.h"
 
 

 
 
 

 
# 11

# 1 "/usr/include/sys/regset.h"
 
 

 
 
 

 
# 12

# 15

#ident	"@(#)regset.h	1.29	07/09/06 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 19 "/usr/include/sys/regset.h"

# 1 "/usr/include/sys/int_types.h"
 
# 5

# 23 "/usr/include/sys/regset.h"

# 27

 
# 74

 
# 91

# 93

# 95
typedef long	greg_t;
# 99

# 106

typedef greg_t	gregset_t[ 21 ];

# 118

# 398

 
# 423

 
# 429

# 431
 
# 441

# 443

 
# 449

struct	__rwindow {
	greg_t	__rw_local[8];		 
	greg_t	__rw_in[8];		 
};

# 457

struct __gwindows {
	int		__wbcnt;
	greg_t		*__spbuf[ 31 ];
	struct __rwindow	__wbuf[ 31 ];
};

typedef struct __gwindows	gwindows_t;

 
# 470

struct __fpq {
	unsigned int *__fpq_addr;	 
	unsigned int __fpq_instr;	 
};

struct __fq {
	union {				 
		double __whole;
		struct __fpq __fpq;
	} _FQu;
};

 
# 486

 
# 496

# 498

 
# 502

struct __fpu {
	union {					 
		uint32_t	__fpu_regs[32];		 
		double		__fpu_dregs[32];	 
		long double	__fpu_qregs[16];	 
	} __fpu_fr;
	struct __fq	*__fpu_q;		 
	uint64_t	__fpu_fsr;	 
	uint8_t		__fpu_qcnt;		 
	uint8_t		__fpu_q_entrysize;	 
	uint8_t		__fpu_en;		 
};

# 535

typedef struct __fpu	fpregset_t;

 
# 542
typedef struct {
	unsigned int	__xrs_id;	 
	caddr_t		__xrs_ptr;	 
} xrs_t;

# 548

# 550

 
# 558
typedef	int64_t	asrset_t[16];	 

# 561

 
# 565
typedef struct {
	gregset_t	__gregs;  
	gwindows_t	*__gwins;  
	fpregset_t	__fpregs;  
	xrs_t		__xrs;	 
# 571
	asrset_t	__asrs;		 
	long		__filler[4];	 
# 576
} mcontext_t;

# 580


# 585

# 25 "/usr/include/sys/ucontext.h"

# 29

 
# 35

# 43

# 54

# 58
typedef	struct __ucontext ucontext_t;
# 60

# 64
struct	__ucontext {
# 66
	uint_t		uc_flags;
	ucontext_t	*uc_link;
	sigset_t   	uc_sigmask;
	stack_t 	uc_stack;
	mcontext_t	uc_mcontext;
# 72
	long		uc_filler[4];
# 76
};

# 98

# 128

# 137

# 141

# 233 "/usr/include/sys/signal.h"

# 311

# 315

# 24 "/usr/include/sys/procset.h"

 
# 34


 
# 41
typedef enum
# 45
		{
	P_PID,		 
	P_PPID,		 
	P_PGID,		 
			 
	P_SID,		 
	P_CID,		 
	P_UID,		 
	P_GID,		 
	P_ALL,		 
	P_LWPID,	 
	P_TASKID,	 
	P_PROJID,	 
	P_POOLID,	 
	P_ZONEID,	 
	P_CTID,		 
	P_CPUID,	 
	P_PSETID	 
} idtype_t;


 
# 122

# 136

# 140

# 27 "/usr/include/sys/wait.h"

# 31

 
# 35

# 38


# 48

 
# 52

# 54

# 59

# 63

# 66

# 68

# 77


# 81

extern pid_t wait(int *);
extern pid_t waitpid(pid_t, int *, int);

# 86
extern int waitid(idtype_t, id_t, siginfo_t *, int);
 
# 92

# 96

# 115

# 119

# 23 "/usr/include/stdlib.h"

 
# 61

# 65

# 74

# 76

 
# 87

 
# 98

# 102
extern int rand_r(unsigned int *);
# 104

extern void _exithandle(void);

# 110
extern double drand48(void);
extern double erand48(unsigned short *);
extern long jrand48(unsigned short *);
extern void lcong48(unsigned short *);
extern long lrand48(void);
extern long mrand48(void);
extern long nrand48(unsigned short *);
extern unsigned short *seed48(unsigned short *);
extern void srand48(long);
extern int putenv(char *);
extern void setkey(const char *);
# 122

 
# 145

# 149
extern int	mkstemp(char *);
# 151

# 156

# 160
extern long a64l(const char *);
extern char *ecvt(double, int, int * restrict , int * restrict );
extern char *fcvt(double, int, int * restrict , int * restrict );
extern char *gcvt(double, int, char *);
extern int getsubopt(char **, char *const *, char **);
extern int  grantpt(int);
extern char *initstate(unsigned, char *, size_t);
extern char *l64a(long);
extern char *mktemp(char *);
extern char *ptsname(int);
extern long random(void);
extern char *realpath(const char * restrict , char * restrict );
extern char *setstate(const char *);
extern void srandom(unsigned);
extern int  unlockpt(int);
 
# 181

# 185
extern int posix_openpt(int);
extern int setenv(const char *, const char *, int);
extern int unsetenv(const char *);
# 189

# 218

# 314

# 318

# 1 "/usr/include/string.h"
 
 

 
 
 

 
# 12

# 15

#ident	"@(#)string.h	1.27	07/01/14 SMI"

# 1 "/usr/include/iso/string_iso.h"
 
 

 
 
 

 
# 12

 
# 24

# 27

#ident	"@(#)string_iso.h	1.5	04/06/18 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 31 "/usr/include/iso/string_iso.h"

# 35

# 39

# 48

# 56

# 58

extern int memcmp(const void *, const void *, size_t);
extern void *memcpy(void * restrict , const void * restrict , size_t);
extern void *memmove(void *, const void *, size_t);
extern void *memset(void *, int, size_t);
extern char *strcat(char * restrict , const char * restrict );
extern int strcmp(const char *, const char *);
extern char *strcpy(char * restrict , const char * restrict );
extern int strcoll(const char *, const char *);
extern size_t strcspn(const char *, const char *);
extern char *strerror(int);
extern size_t strlen(const char *);
extern char *strncat(char * restrict , const char * restrict , size_t);
extern int strncmp(const char *, const char *, size_t);
extern char *strncpy(char * restrict , const char * restrict , size_t);
extern size_t strspn(const char *, const char *);
extern char *strtok(char * restrict , const char * restrict );
extern size_t strxfrm(char * restrict , const char * restrict , size_t);

 
# 129
extern void *memchr(const void *, int, size_t);
extern char *strchr(const char *, int);
extern char *strpbrk(const char *, const char *);
extern char *strrchr(const char *, int);
extern char *strstr(const char *, const char *);
# 135

# 162

# 166

# 170

# 19 "/usr/include/string.h"

 
# 49

# 53

# 55

# 59
extern int strerror_r(int, char *, size_t);
# 61

# 65
extern char *strtok_r(char * restrict , const char * restrict ,
	char ** restrict );
# 68

# 71
extern void *memccpy(void * restrict , const void * restrict ,
		int, size_t);
# 74

# 86

# 90
extern char *strdup(const char *);
# 92

# 126

# 130

# 1 "/usr/include/locale.h"
 
# 5

# 8

#ident	"@(#)locale.h	1.20	03/12/04 SMI"

 
# 25
 
# 28
 
# 32
 
# 45

# 1 "/usr/include/iso/locale_iso.h"
 
# 5

 
# 20

 
# 29

 
# 41

# 44

#ident	"@(#)locale_iso.h	1.3	03/12/04 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 48 "/usr/include/iso/locale_iso.h"

# 52

# 56

struct lconv {
	char *decimal_point;	 
	char *thousands_sep;	 
	char *grouping;			 
	char *int_curr_symbol;	 
	char *currency_symbol;	 
	char *mon_decimal_point;	 
	char *mon_thousands_sep;	 
	char *mon_grouping;		 
	char *positive_sign;	 
	char *negative_sign;	 
	char int_frac_digits;	 
	char frac_digits;		 
	char p_cs_precedes;		 
	char p_sep_by_space;	 
	char n_cs_precedes;		 
	char n_sep_by_space;	 
	char p_sign_posn;		 
	char n_sign_posn;		 

 
# 84
	char int_p_cs_precedes;		 
	char int_p_sep_by_space;	 
	char int_n_cs_precedes;		 
	char int_n_sep_by_space;	 
	char int_p_sign_posn;		 
	char int_n_sign_posn;		 
# 91
};

# 100

# 108

# 110
extern char	*setlocale(int, const char *);
extern struct lconv *localeconv(void);
# 116

# 120

# 124

# 47 "/usr/include/locale.h"

# 52

 
# 62

# 66

# 68

# 72

# 76

# 1 "/usr/include/sys/utsname.h"
 
 

 
 
 

 
# 12

# 15

#ident	"@(#)utsname.h	1.30	04/09/28 SMI"

# 1 "/usr/include/sys/feature_tests.h"
 
# 4

# 19 "/usr/include/sys/utsname.h"

# 23

# 25
				 
				 

# 33

struct utsname {
	char	sysname[ 257 ];
	char	nodename[ 257 ];
	char	release[ 257 ];
	char	version[ 257 ];
	char	machine[ 257 ];
};

# 45

# 47

# 106

# 108
extern int uname(struct utsname *);
# 112

# 114

# 121

# 125

# 20 "argv_dump.c"

int main(int argc, char *argv[]) 
{

    int foo = 0;
    int char_count_total = 0;
    struct utsname uname_data;

    setlocale(  5 , "C" );
    if ( uname( &uname_data ) < 0 ) {
        fprintf (  ( & __iob [ 2 ] ),
                 "WARNING : Could not attain system uname data.\n" );
        perror ( "uname" );
    } else {
        printf ( "-------------------------------" );
        printf ( "------------------------------\n" );
        printf ( "        system name = %s\n", uname_data.sysname );
        printf ( "          node name = %s\n", uname_data.nodename );
        printf ( "            release = %s\n", uname_data.release );
        printf ( "            version = %s\n", uname_data.version );
        printf ( "            machine = %s\n", uname_data.machine );
        printf ( "-------------------------------" );
        printf ( "------------------------------" );
    }
    printf ("\n");

     
# 52

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

    return ( 42 );

}

#ident "acomp: Studio 12.6 Sun C 5.15 SunOS_sparc 2017/05/30"
