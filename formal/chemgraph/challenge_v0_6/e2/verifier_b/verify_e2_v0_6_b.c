#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "json.h"
#include "sha256.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

#define Q16 ((int64_t)65536)
#define RAW_SIDE 768
#define PAD_SIDE 1024
#define RAW_BYTES ((size_t)RAW_SIDE * RAW_SIDE)
#define PAD_BYTES ((size_t)PAD_SIDE * PAD_SIDE)
#define PAIR_COUNT 8
#define TOTAL_TUPLES 36088416

__extension__ typedef __int128 int128_t;

static const char *V06_SHA = "e326fc744903b998793746f299232c4225ce3b8648e1abdda7b4e065f06a9402";
static const char *V05_SHA = "55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65";
static const char *GRAPH_LEFT = "{\"edges\":[[0,1,1],[2,3,1]],\"nodes\":[[\"C\",0],[\"C\",0],[\"C\",0],[\"C\",0]]}";
static const char *GRAPH_RIGHT = "{\"edges\":[[0,1,1]],\"nodes\":[[\"C\",0],[\"C\",0],[\"TEXT_STROKE\",0]]}";

typedef struct { unsigned char *data; size_t length; } Bytes;
typedef struct { int64_t x0,y0,x1,y1; } Segment;
typedef struct {
    int half, motif, width, cap, aa, reflection, rotation, scale;
    int64_t tuple_id;
} Tuple;
typedef struct {
    char pair_id[128], cap_name[16]; int width;
    int64_t left_tuple_id,right_tuple_id;
    char left_raw_sha[65],right_raw_sha[65],left_padded_sha[65],right_padded_sha[65];
    char left_gray_sha[65], right_gray_sha[65];
    char left_png_sha[65], right_png_sha[65];
    char left_graph_sha[65],right_graph_sha[65];
    size_t png_length;
} PairResult;
typedef struct { char text[1024]; int failed; } Failure;
typedef struct {
    int ok;
    size_t count;
    uint64_t artifact_bytes;
    unsigned seen[2][4];
    PairResult pairs[2];
    Failure failure;
} WorkerReport;

static void set_failure(Failure *f, const char *fmt, ...) {
    va_list ap;
    if (f->failed) return;
    f->failed = 1; va_start(ap, fmt);
    vsnprintf(f->text, sizeof(f->text), fmt, ap); va_end(ap);
}

static int lowercase_hex64(const char *value) {
    size_t i;
    if(!value||strlen(value)!=64)return 0;
    for(i=0;i<64;++i)if(!((value[i]>='0'&&value[i]<='9')||(value[i]>='a'&&value[i]<='f')))return 0;
    return 1;
}

static int read_file(const char *path, size_t maximum, Bytes *out, Failure *f) {
    FILE *fp; struct stat st; size_t got;
    memset(out, 0, sizeof(*out));
    if (stat(path, &st) || st.st_size < 0 || (uintmax_t)st.st_size > maximum) {
        set_failure(f, "cannot stat or oversized file: %s", path); return 0;
    }
    fp = fopen(path, "rb");
    if (!fp) { set_failure(f, "cannot open file: %s", path); return 0; }
    out->length = (size_t)st.st_size;
    out->data = malloc(out->length ? out->length : 1);
    if (!out->data) { fclose(fp); set_failure(f, "out of memory reading %s", path); return 0; }
    got = fread(out->data, 1, out->length, fp);
    if (got != out->length || fclose(fp)) {
        free(out->data); memset(out,0,sizeof(*out)); set_failure(f,"short read: %s",path); return 0;
    }
    return 1;
}

static int file_hash_is(const char *path, const char *expected, Failure *f) {
    Bytes b; char actual[65];
    if (!read_file(path, 16 * 1024 * 1024, &b, f)) return 0;
    sha256_hex(b.data,b.length,actual); free(b.data);
    if (strcmp(actual,expected)) { set_failure(f,"frozen contract hash mismatch: %s",path); return 0; }
    return 1;
}

static int64_t round_div_signed(int64_t numerator, int64_t denominator) {
    int negative = numerator < 0; uint64_t n = negative ? (uint64_t)(-numerator) : (uint64_t)numerator;
    uint64_t q = n / (uint64_t)denominator, r = n % (uint64_t)denominator;
    if (r * 2 > (uint64_t)denominator || (r * 2 == (uint64_t)denominator && (q & 1))) ++q;
    return negative ? -(int64_t)q : (int64_t)q;
}

static uint64_t isqrt_u64(uint64_t n) {
    uint64_t result = 0, bit = (uint64_t)1 << 62;
    while (bit > n) bit >>= 2;
    while (bit) {
        if (n >= result + bit) { n -= result + bit; result = (result >> 1) + bit; }
        else result >>= 1;
        bit >>= 2;
    }
    return result;
}

static int tuple_encode(Tuple *t, Failure *f) {
    int64_t id;
    if (t->half < 60 || t->half > 110 || t->motif < 0 || t->motif > 51 ||
        t->width < 1 || t->width > 9 || t->cap < 0 || t->cap > 2 ||
        t->aa < 0 || t->aa > 2 || t->reflection < 0 || t->reflection > 2 ||
        t->rotation < 0 || t->rotation > 7 || t->scale < 0 || t->scale > 6) {
        set_failure(f,"tuple axis outside frozen domain"); return 0;
    }
    id=t->half-60; id=id*52+t->motif; id=id*9+(t->width-1); id=id*3+t->cap;
    id=id*3+t->aa; id=id*3+t->reflection; id=id*8+t->rotation; id=id*7+t->scale;
    if (id < 0 || id >= TOTAL_TUPLES || id != t->tuple_id) {
        set_failure(f,"tuple ID does not match frozen mixed-radix rank"); return 0;
    }
    return 1;
}

static int object_int(const Json *o, const char *key, int *out, Failure *f) {
    int64_t v;
    if (!json_int(json_get(o,key),&v) || v < INT_MIN || v > INT_MAX) {
        set_failure(f,"missing or invalid integer field %s",key); return 0;
    }
    *out=(int)v; return 1;
}

static int object_i64(const Json *o, const char *key, int64_t *out, Failure *f) {
    if (!json_int(json_get(o,key),out)) { set_failure(f,"missing or invalid integer field %s",key); return 0; }
    return 1;
}

static int string_index(const char *value, const char *const *items, int count) {
    int i; if (!value) return -1;
    for(i=0;i<count;++i) if(!strcmp(value,items[i])) return i;
    return -1;
}

static int parse_tuple(const Json *o, Tuple *t, Failure *f) {
    static const char *const caps[]={"butt","round","square"};
    static const char *const aas[]={"binary_1x","box_2x2","box_4x4"};
    static const char *const refs[]={"none","x","y"};
    const char *s;
    if (!o || o->kind != JSON_OBJECT) { set_failure(f,"tuple is not an object"); return 0; }
    if (!object_i64(o,"tuple_id",&t->tuple_id,f) || !object_int(o,"half_length",&t->half,f) ||
        !object_int(o,"motif_ordinal",&t->motif,f) || !object_int(o,"width_px",&t->width,f) ||
        !object_int(o,"rotation_index",&t->rotation,f) || !object_int(o,"scale_index",&t->scale,f)) return 0;
    s=json_string(json_get(o,"cap")); t->cap=string_index(s,caps,3);
    s=json_string(json_get(o,"antialias_profile")); t->aa=string_index(s,aas,3);
    s=json_string(json_get(o,"reflection")); t->reflection=string_index(s,refs,3);
    return tuple_encode(t,f);
}

static int tuple_matches(const Tuple *t, int side, int cap, int width) {
    if(t->width!=width || t->cap!=cap || t->aa!=0 || t->reflection!=0) return 0;
    if(!side) return t->half==62 && t->motif==11 && t->rotation==0 && t->scale==2;
    return t->half==93 && t->motif==48 && t->rotation==4 && t->scale==0;
}

static void transform_point(int64_t x, int64_t y, const Tuple *t, int center,
                            int64_t *ox, int64_t *oy) {
    static const int cos_num[8]={1,3,5,8,0,-3,5,-8};
    static const int cos_den[8]={1,5,13,17,1,5,13,17};
    static const int sin_num[8]={0,4,12,15,1,4,-12,-15};
    static const int sin_den[8]={1,5,13,17,1,5,13,17};
    static const int scale_num[7]={1,5,3,1,5,3,2};
    static const int scale_den[7]={2,8,4,1,4,2,1};
    int64_t common, rx, ry;
    if(t->reflection==1)y=-y; else if(t->reflection==2)x=-x;
    common=(int64_t)cos_den[t->rotation]*sin_den[t->rotation];
    rx=round_div_signed(x*cos_num[t->rotation]*sin_den[t->rotation]-y*sin_num[t->rotation]*cos_den[t->rotation],common);
    ry=round_div_signed(x*sin_num[t->rotation]*cos_den[t->rotation]+y*cos_num[t->rotation]*sin_den[t->rotation],common);
    *ox=round_div_signed(rx*scale_num[t->scale],scale_den[t->scale])+(int64_t)center*Q16;
    *oy=round_div_signed(ry*scale_num[t->scale],scale_den[t->scale])+(int64_t)center*Q16;
}

static void make_segments(const Tuple *t, int center, Segment out[2]) {
    int coords[2][4]; int i;
    if(t->motif==11) {
        coords[0][0]=-t->half; coords[0][1]=0; coords[0][2]=-15; coords[0][3]=0;
        coords[1][0]=15; coords[1][1]=-t->half; coords[1][2]=15; coords[1][3]=t->half;
    } else {
        coords[0][0]=-t->half; coords[0][1]=-80; coords[0][2]=t->half; coords[0][3]=-80;
        coords[1][0]=0; coords[1][1]=-35; coords[1][2]=0; coords[1][3]=35;
    }
    for(i=0;i<2;++i){
        transform_point((int64_t)coords[i][0]*Q16,(int64_t)coords[i][1]*Q16,t,center,&out[i].x0,&out[i].y0);
        transform_point((int64_t)coords[i][2]*Q16,(int64_t)coords[i][3]*Q16,t,center,&out[i].x1,&out[i].y1);
    }
}

static int inside_segment(int64_t sx, int64_t sy, const Segment *s, const Tuple *t) {
    int64_t vx=s->x1-s->x0, vy=s->y1-s->y0, wx=sx-s->x0, wy=sy-s->y0;
    int128_t length_sq=(int128_t)vx*vx+(int128_t)vy*vy;
    int128_t projection=(int128_t)wx*vx+(int128_t)wy*vy;
    int128_t cross=(int128_t)wx*vy-(int128_t)wy*vx;
    int64_t radius=(int64_t)t->width*Q16/2;
    int128_t rhs=(int128_t)radius*radius*length_sq;
    if(length_sq==0)return 0;
    if(t->cap==0)return projection>=0 && projection<=length_sq && cross*cross<=rhs;
    if(t->cap==2){
        uint64_t len=isqrt_u64((uint64_t)length_sq);
        int128_t extension=(int128_t)radius*len;
        return projection>=-extension && projection<=length_sq+extension && cross*cross<=rhs;
    }
    return 0;
}

static int floor_q16(int64_t x) {
    if(x>=0)return (int)(x/Q16);
    return (int)(-(((-x)+Q16-1)/Q16));
}

static Bytes rasterize(const Tuple *t, int side, Failure *f) {
    Segment segments[2]; Bytes result={0}; int si,x,y,padding,minx,maxx,miny,maxy;
    result.length=(size_t)side*side; result.data=malloc(result.length);
    if(!result.data){set_failure(f,"out of memory rasterizing");return result;}
    memset(result.data,255,result.length); make_segments(t,side/2,segments);
    padding=t->width+2;
    for(si=0;si<2;++si){
        minx=floor_q16(segments[si].x0<segments[si].x1?segments[si].x0:segments[si].x1)-padding;
        maxx=floor_q16(segments[si].x0>segments[si].x1?segments[si].x0:segments[si].x1)+padding;
        miny=floor_q16(segments[si].y0<segments[si].y1?segments[si].y0:segments[si].y1)-padding;
        maxy=floor_q16(segments[si].y0>segments[si].y1?segments[si].y0:segments[si].y1)+padding;
        if(minx<0)minx=0;
        if(miny<0)miny=0;
        if(maxx>=side)maxx=side-1;
        if(maxy>=side)maxy=side-1;
        for(y=miny;y<=maxy;++y)for(x=minx;x<=maxx;++x)
            if(inside_segment((int64_t)x*Q16+Q16/2,(int64_t)y*Q16+Q16/2,&segments[si],t))
                result.data[(size_t)y*side+x]=0;
    }
    return result;
}

static int bbox(const Bytes *gray, int side, int *minx,int *miny,int *maxx,int *maxy) {
    int x,y; *minx=side;*miny=side;*maxx=-1;*maxy=-1;
    for(y=0;y<side;++y)for(x=0;x<side;++x)if(gray->data[(size_t)y*side+x]!=255){
        if(x<*minx)*minx=x;
        if(x>*maxx)*maxx=x;
        if(y<*miny)*miny=y;
        if(y>*maxy)*maxy=y;
    }
    return *maxx>=0;
}

static int padded_certificate(const Bytes *raw,const Bytes *pad,int margins[4],Failure *f) {
    int x,y,minx,miny,maxx,maxy;
    if(raw->length!=RAW_BYTES||pad->length!=PAD_BYTES){set_failure(f,"padded certificate length mismatch");return 0;}
    for(y=0;y<PAD_SIDE;++y)for(x=0;x<PAD_SIDE;++x){
        unsigned char p=pad->data[(size_t)y*PAD_SIDE+x];
        if(x>=128&&x<896&&y>=128&&y<896){
            if(p!=raw->data[(size_t)(y-128)*RAW_SIDE+(x-128)]){set_failure(f,"padded embedded bytes differ");return 0;}
        } else if(p!=255){set_failure(f,"padded foreground outside embedded rectangle");return 0;}
    }
    if(!bbox(raw,RAW_SIDE,&minx,&miny,&maxx,&maxy)){set_failure(f,"empty foreground");return 0;}
    margins[0]=minx;margins[1]=miny;margins[2]=RAW_SIDE-1-maxx;margins[3]=RAW_SIDE-1-maxy;
    if(margins[0]<1||margins[1]<1||margins[2]<1||margins[3]<1){set_failure(f,"raw white border missing");return 0;}
    return 1;
}

static Bytes canonicalize(const Bytes *raw, Failure *f) {
    Bytes c={0}; int minx,miny,maxx,maxy,w,h,left,top,y;
    if(!bbox(raw,RAW_SIDE,&minx,&miny,&maxx,&maxy)){set_failure(f,"empty foreground");return c;}
    w=maxx-minx+1;h=maxy-miny+1;left=(RAW_SIDE-w)/2;top=(RAW_SIDE-h)/2;
    c.length=RAW_BYTES;c.data=malloc(c.length);if(!c.data){set_failure(f,"out of memory canonicalizing");return c;}
    memset(c.data,255,c.length);
    for(y=0;y<h;++y)memcpy(c.data+(size_t)(top+y)*RAW_SIDE+left,
                           raw->data+(size_t)(miny+y)*RAW_SIDE+minx,(size_t)w);
    return c;
}

static void put_be32(unsigned char *p,uint32_t x){p[0]=x>>24;p[1]=x>>16;p[2]=x>>8;p[3]=x;}

static int append_chunk(unsigned char *png,size_t cap,size_t *cursor,const char kind[4],
                        const unsigned char *payload,size_t length) {
    uLong crc;
    if(length>UINT32_MAX||*cursor+12+length>cap)return 0;
    put_be32(png+*cursor,(uint32_t)length);memcpy(png+*cursor+4,kind,4);
    if(length)memcpy(png+*cursor+8,payload,length);
    crc=crc32(0L,Z_NULL,0);crc=crc32(crc,(const Bytef*)kind,4);
    if(length)crc=crc32(crc,payload,(uInt)length);
    put_be32(png+*cursor+8+length,(uint32_t)crc);*cursor+=12+length;return 1;
}

static Bytes encode_png(const Bytes *gray, Failure *f) {
    static const unsigned char signature[8]={137,80,78,71,13,10,26,10};
    Bytes png={0}; unsigned char ihdr[13]={0}; unsigned char *scan=NULL,*compressed=NULL;
    size_t scan_len=(RAW_SIDE+1u)*RAW_SIDE,cursor=0;uLongf compressed_len=compressBound(scan_len);int y,z;
    scan=malloc(scan_len);compressed=malloc(compressed_len);
    if(!scan||!compressed){set_failure(f,"out of memory encoding PNG");goto done;}
    for(y=0;y<RAW_SIDE;++y){scan[(size_t)y*(RAW_SIDE+1)]=0;memcpy(scan+(size_t)y*(RAW_SIDE+1)+1,gray->data+(size_t)y*RAW_SIDE,RAW_SIDE);}
    z=compress2(compressed,&compressed_len,scan,scan_len,9);
    if(z!=Z_OK){set_failure(f,"zlib canonical PNG encoding failed");goto done;}
    png.length=8+25+12+compressed_len+12;png.data=malloc(png.length);
    if(!png.data){set_failure(f,"out of memory assembling PNG");png.length=0;goto done;}
    memcpy(png.data,signature,8);cursor=8;put_be32(ihdr,RAW_SIDE);put_be32(ihdr+4,RAW_SIDE);ihdr[8]=8;
    if(!append_chunk(png.data,png.length,&cursor,"IHDR",ihdr,13)||
       !append_chunk(png.data,png.length,&cursor,"IDAT",compressed,compressed_len)||
       !append_chunk(png.data,png.length,&cursor,"IEND",NULL,0)||cursor!=png.length){
        free(png.data);png=(Bytes){0};set_failure(f,"PNG assembly invariant failed");
    }
done: free(scan);free(compressed);return png;
}

static int json_array_ints(const Json *v, const int *expected, size_t count) {
    size_t i; int64_t n;
    if(!v||v->kind!=JSON_ARRAY||v->as.array.length!=count)return 0;
    for(i=0;i<count;++i)if(!json_int(v->as.array.items[i],&n)||n!=expected[i])return 0;
    return 1;
}

static int within_root(const char *root,const char *relative,char resolved[PATH_MAX],Failure *f) {
    char root_real[PATH_MAX],candidate[PATH_MAX];size_t n;
    if(!relative||relative[0]=='/'||strstr(relative,"//")){
        set_failure(f,"artifact path must be a normalized relative path");return 0;
    }
    {
        const char *p=relative;
        while(*p){const char *slash=strchr(p,'/');size_t len=slash?(size_t)(slash-p):strlen(p);
            if((len==1&&p[0]=='.')||(len==2&&p[0]=='.'&&p[1]=='.')){
                set_failure(f,"artifact path traversal is forbidden");return 0;
            }
            if(!slash)break;
            p=slash+1;
        }
    }
    if(!realpath(root,root_real)){set_failure(f,"cannot resolve artifact root");return 0;}
    if(snprintf(candidate,sizeof(candidate),"%s/%s",root_real,relative)>=(int)sizeof(candidate)||
       !realpath(candidate,resolved)){set_failure(f,"cannot resolve artifact path: %s",relative);return 0;}
    n=strlen(root_real);
    if(strncmp(root_real,resolved,n)||resolved[n]!='/'){
        set_failure(f,"artifact resolves outside root: %s",relative);return 0;
    }
    return 1;
}

static int verify_descriptor(const Json *d,const char *root,const char *format,int width,int height,
                             const Bytes *expected,Bytes *actual,Failure *f) {
    const char *path,*declared_format,*declared_sha;int64_t declared_length,w,h;char resolved[PATH_MAX],actual_sha[65];
    if(!d||d->kind!=JSON_OBJECT){set_failure(f,"artifact descriptor is not an object");return 0;}
    path=json_string(json_get(d,"path"));declared_format=json_string(json_get(d,"format"));
    declared_sha=json_string(json_get(d,"sha256"));
    if(!path||!declared_format||!lowercase_hex64(declared_sha)||
       !json_int(json_get(d,"byte_length"),&declared_length)||
       !json_int(json_get(d,"width"),&w)||!json_int(json_get(d,"height"),&h)){
        set_failure(f,"invalid artifact descriptor fields");return 0;
    }
    if(strcmp(declared_format,format)||w!=width||h!=height){set_failure(f,"artifact descriptor profile mismatch: %s",path);return 0;}
    if(!within_root(root,path,resolved,f)||!read_file(resolved,2ULL*1024*1024*1024,actual,f))return 0;
    sha256_hex(actual->data,actual->length,actual_sha);
    if(declared_length<0||(uint64_t)declared_length!=actual->length||strcmp(declared_sha,actual_sha)){
        set_failure(f,"artifact descriptor commitment mismatch: %s",path);return 0;
    }
    if(actual->length!=expected->length||memcmp(actual->data,expected->data,expected->length)){
        set_failure(f,"artifact differs from independently reconstructed full bytes: %s",path);return 0;
    }
    return 1;
}

static int verify_certificate(const Json *cert,const Bytes *raw,int margins[4],Failure *f) {
    int yes,bbox_values[4],offset[2]={128,128};
    if(!cert||cert->kind!=JSON_OBJECT){set_failure(f,"missing padded clipping certificate");return 0;}
    if(!json_bool(json_get(cert,"passed"),&yes)||!yes||
       !json_bool(json_get(cert,"embedded_raw_full_bytes_equal"),&yes)||!yes||
       !json_bool(json_get(cert,"outer_white"),&yes)||!yes||
       !json_array_ints(json_get(cert,"offset"),offset,2)){
        set_failure(f,"padded clipping certificate claims are invalid");return 0;
    }
    bbox_values[0]=margins[0]+128;bbox_values[1]=margins[1]+128;
    bbox_values[2]=RAW_SIDE-1-margins[2]+128;bbox_values[3]=RAW_SIDE-1-margins[3]+128;
    if(!json_array_ints(json_get(cert,"foreground_bbox"),bbox_values,4)){
        set_failure(f,"padded clipping certificate bbox differs from reconstruction");return 0;
    }
    (void)raw;return 1;
}

static int verify_graph(const Json *graph,const char *canonical,int nodes,int edges,const char *decode,Failure *f) {
    const char *bytes,*sha,*kind;int64_t n,e,len;char actual_sha[65];size_t canonical_len=strlen(canonical);
    if(!graph||graph->kind!=JSON_OBJECT){set_failure(f,"missing graph signature");return 0;}
    bytes=json_string(json_get(graph,"canonical_bytes_ascii"));sha=json_string(json_get(graph,"sha256"));
    kind=json_string(json_get(graph,"decode_kind"));
    if(!bytes||!sha||!kind||!json_int(json_get(graph,"byte_length"),&len)||
       !json_int(json_get(graph,"node_count"),&n)||!json_int(json_get(graph,"edge_count"),&e)){
        set_failure(f,"invalid graph signature fields");return 0;
    }
    sha256_hex(canonical,canonical_len,actual_sha);
    if(strcmp(bytes,canonical)||len!=(int64_t)canonical_len||strcmp(sha,actual_sha)||n!=nodes||e!=edges||strcmp(kind,decode)){
        set_failure(f,"graph signature differs from independent semantic construction");return 0;
    }
    return 1;
}

static const Json *require_object(const Json *o,const char *key,Failure *f) {
    const Json *v=json_get(o,key);
    if(!v||v->kind!=JSON_OBJECT)set_failure(f,"missing object field %s",key);
    return v;
}

static int side_interpretation_zero(const Json *side,Failure *f) {
    const Json *v=json_get(side,"interpretation");int64_t index;int member;
    if(json_int(json_get(side,"interpretation_index"),&index)&&index==0&&
       json_bool(json_get(side,"interpretation_membership"),&member)&&member)return 1;
    if(v&&v->kind==JSON_INT)return v->as.integer==0;
    if(v&&v->kind==JSON_OBJECT&&json_int(json_get(v,"interpretation_index"),&index)&&index==0)return 1;
    set_failure(f,"selected interpretation must be index 0");return 0;
}

static int verify_side(const Json *side,const char *root,int side_index,int cap,int width,
                       Tuple *tuple,Bytes *raw,Bytes *pad,Bytes *canonical,Bytes *png,
                       int margins[4],Failure *f) {
    const Json *tuple_json,*membership;const char *family;int member;
    Bytes raw_file={0},pad_file={0},canonical_file={0},png_file={0};
    if(!side||side->kind!=JSON_OBJECT){set_failure(f,"pair side is not an object");return 0;}
    tuple_json=require_object(side,"tuple",f);if(f->failed)return 0;
    if(!parse_tuple(tuple_json,tuple,f)||!tuple_matches(tuple,side_index,cap,width)){
        if(!f->failed)set_failure(f,"tuple does not match preregistered E2 witness plan");
        return 0;
    }
    membership=json_get(side,"tuple_membership");
    if(!json_bool(membership,&member)||!member){set_failure(f,"manifest membership field is not true");return 0;}
    family=json_string(json_get(side,"family"));
    if(!family||strcmp(family,side_index?"c4":"s8")){set_failure(f,"motif family mismatch");return 0;}
    if(!side_interpretation_zero(side,f))return 0;
    *raw=rasterize(tuple,RAW_SIDE,f);*pad=rasterize(tuple,PAD_SIDE,f);
    if(f->failed||!padded_certificate(raw,pad,margins,f))return 0;
    *canonical=canonicalize(raw,f);if(f->failed)return 0;
    *png=encode_png(canonical,f);if(f->failed)return 0;
    if(!verify_descriptor(json_get(side,"raw_gray"),root,"gray8-row-major",RAW_SIDE,RAW_SIDE,raw,&raw_file,f)||
       !verify_descriptor(json_get(side,"padded_gray"),root,"gray8-row-major",PAD_SIDE,PAD_SIDE,pad,&pad_file,f)||
       !verify_descriptor(json_get(side,"canonical_gray"),root,"gray8-row-major",RAW_SIDE,RAW_SIDE,canonical,&canonical_file,f)||
       !verify_descriptor(json_get(side,"canonical_png"),root,"canonical-gray8-png",RAW_SIDE,RAW_SIDE,png,&png_file,f)||
       !verify_certificate(json_get(side,"padded_clipping_certificate"),raw,margins,f)||
       !verify_graph(json_get(side,"graph"),side_index?GRAPH_RIGHT:GRAPH_LEFT,
                     side_index?3:4,side_index?1:2,side_index?"stroke_role_ambiguity":"strict_graph",f)) goto done;
done:
    free(raw_file.data);free(pad_file.data);free(canonical_file.data);free(png_file.data);
    return !f->failed;
}

static int require_binding(const Json *root,Failure *f) {
    const Json *b=require_object(root,"contract_binding",f);const char *p,*v;int64_t count;
    if(f->failed)return 0;
    p=json_string(json_get(b,"parent_contract_sha256"));v=json_string(json_get(b,"v0_6_contract_sha256"));
    if(!p||!v||strcmp(p,V05_SHA)||strcmp(v,V06_SHA)||
       !json_int(json_get(b,"parent_domain_cardinality"),&count)||count!=TOTAL_TUPLES){
        set_failure(f,"manifest contract binding mismatch");return 0;
    }
    return 1;
}

static int verify_checks(const Json *checks,int raw_distinct,int gray_equal,int png_equal,Failure *f) {
    int b;
    if(!checks||checks->kind!=JSON_OBJECT||
       !json_bool(json_get(checks,"raw_gray_full_bytes_distinct"),&b)||b!=raw_distinct||
       !json_bool(json_get(checks,"canonical_gray_full_bytes_equal"),&b)||b!=gray_equal||
       !json_bool(json_get(checks,"canonical_png_full_bytes_equal"),&b)||b!=png_equal||
       !json_bool(json_get(checks,"graph_signatures_distinct"),&b)||!b||
       !json_bool(json_get(checks,"padded_clipping_certificates_passed"),&b)||!b||
       !json_bool(json_get(checks,"tuple_ids_distinct"),&b)||!b){
        set_failure(f,"pair checks differ from full-byte reconstruction");return 0;
    }
    return 1;
}

static int verify_pair(const Json *pair,const char *root,unsigned seen[2][4],PairResult *result,Failure *f) {
    const Json *left,*right,*style;const char *id,*style_cap,*style_aa;Tuple lt,rt;Bytes lr={0},lp={0},lc={0},li={0},rr={0},rp={0},rc={0},ri={0};
    int lm[4],rm[4],raw_distinct,gray_equal,png_equal,cap,width,wi;
    if(!pair||pair->kind!=JSON_OBJECT){set_failure(f,"pair is not an object");return 0;}
    id=json_string(json_get(pair,"pair_id"));if(!id||!*id||strlen(id)>=sizeof(result->pair_id)){
        set_failure(f,"invalid pair_id");return 0;
    }
    left=require_object(pair,"left",f);right=require_object(pair,"right",f);if(f->failed)return 0;
    /* Read style from left tuple first, then enforce both sides and the Cartesian plan. */
    if(!parse_tuple(require_object(left,"tuple",f),&lt,f))return 0;
    cap=lt.cap;width=lt.width;wi=(width==2?0:width==4?1:width==6?2:width==8?3:-1);
    if((cap!=0&&cap!=2)||wi<0||seen[cap==2][wi]){set_failure(f,"duplicate or out-of-plan cap/width pair");return 0;}
    style=require_object(pair,"style",f);style_cap=json_string(json_get(style,"cap"));style_aa=json_string(json_get(style,"antialias_profile"));
    {
        int style_width;
        if(f->failed||!object_int(style,"width_px",&style_width,f)||style_width!=width||!style_cap||
           strcmp(style_cap,cap==0?"butt":"square")||!style_aa||strcmp(style_aa,"binary_1x")){
            if(!f->failed)set_failure(f,"pair style does not match tuples");
            return 0;
        }
    }
    if(!verify_side(left,root,0,cap,width,&lt,&lr,&lp,&lc,&li,lm,f)||
       !verify_side(right,root,1,cap,width,&rt,&rr,&rp,&rc,&ri,rm,f))goto done;
    raw_distinct=lr.length!=rr.length||memcmp(lr.data,rr.data,lr.length)!=0;
    gray_equal=lc.length==rc.length&&memcmp(lc.data,rc.data,lc.length)==0;
    png_equal=li.length==ri.length&&memcmp(li.data,ri.data,li.length)==0;
    if(!raw_distinct||!gray_equal||!png_equal||!strcmp(GRAPH_LEFT,GRAPH_RIGHT)){
        set_failure(f,"E2 full-byte or semantic non-equivalence claim failed");goto done;
    }
    if(!verify_checks(json_get(pair,"checks"),raw_distinct,gray_equal,png_equal,f))goto done;
    seen[cap==2][wi]=1;strcpy(result->pair_id,id);strcpy(result->cap_name,cap==0?"butt":"square");result->width=width;
    result->left_tuple_id=lt.tuple_id;result->right_tuple_id=rt.tuple_id;
    sha256_hex(lr.data,lr.length,result->left_raw_sha);sha256_hex(rr.data,rr.length,result->right_raw_sha);
    sha256_hex(lp.data,lp.length,result->left_padded_sha);sha256_hex(rp.data,rp.length,result->right_padded_sha);
    sha256_hex(lc.data,lc.length,result->left_gray_sha);sha256_hex(rc.data,rc.length,result->right_gray_sha);
    sha256_hex(li.data,li.length,result->left_png_sha);sha256_hex(ri.data,ri.length,result->right_png_sha);
    sha256_hex(GRAPH_LEFT,strlen(GRAPH_LEFT),result->left_graph_sha);
    sha256_hex(GRAPH_RIGHT,strlen(GRAPH_RIGHT),result->right_graph_sha);
    result->png_length=li.length;
done:
    free(lr.data);free(lp.data);free(lc.data);free(li.data);free(rr.data);free(rp.data);free(rc.data);free(ri.data);
    return !f->failed;
}

static int validate_witness_plan(const Json *root,Failure *f) {
    const Json *w=require_object(root,"witness_plan",f),*left,*right;const char *aa;int n;
    int widths[4]={2,4,6,8};
    if(f->failed)return 0;
    aa=json_string(json_get(w,"antialias_profile"));
    if(!aa||strcmp(aa,"binary_1x")||!object_int(w,"witness_pair_count",&n,f)||n!=8||
       !json_array_ints(json_get(w,"width_px"),widths,4)){
        if(!f->failed)set_failure(f,"manifest witness plan mismatch");
        return 0;
    }
    {
        const Json *caps=json_get(w,"caps");
        if(!caps||caps->kind!=JSON_ARRAY||caps->as.array.length!=2||
           !json_string(caps->as.array.items[0])||strcmp(json_string(caps->as.array.items[0]),"butt")||
           !json_string(caps->as.array.items[1])||strcmp(json_string(caps->as.array.items[1]),"square")){
            set_failure(f,"manifest cap plan mismatch");return 0;
        }
    }
    left=require_object(w,"left_fixed",f);right=require_object(w,"right_fixed",f);if(f->failed)return 0;
    {
        int half,motif,interp,rot,scale;const char *reflection;
        if(!object_int(left,"half_length",&half,f)||!object_int(left,"motif_ordinal",&motif,f)||
           !object_int(left,"interpretation_index",&interp,f)||!object_int(left,"rotation_index",&rot,f)||
           !object_int(left,"scale_index",&scale,f)||(reflection=json_string(json_get(left,"reflection")))==NULL||
           half!=62||motif!=11||interp!=0||rot!=0||scale!=2||strcmp(reflection,"none")){
            if(!f->failed)set_failure(f,"left fixed witness plan mismatch");
            return 0;
        }
        if(!object_int(right,"half_length",&half,f)||!object_int(right,"motif_ordinal",&motif,f)||
           !object_int(right,"interpretation_index",&interp,f)||!object_int(right,"rotation_index",&rot,f)||
           !object_int(right,"scale_index",&scale,f)||(reflection=json_string(json_get(right,"reflection")))==NULL||
           half!=93||motif!=48||interp!=0||rot!=4||scale!=0||strcmp(reflection,"none")){
            if(!f->failed)set_failure(f,"right fixed witness plan mismatch");
            return 0;
        }
    }
    return 1;
}

static int validate_scope(const Json *root,Failure *f) {
    const Json *s=require_object(root,"scope",f);const char *claim;int b;
    if(f->failed)return 0;
    claim=json_string(json_get(s,"claim"));
    if(!claim||strcmp(claim,"exact_full_domain_existential_cross_tuple_full_bytes")||
       !json_bool(json_get(s,"training_authorized"),&b)||b||
       !json_bool(json_get(s,"full_domain_collision_rate_claimed"),&b)||b||
       !json_bool(json_get(s,"sampled_to_full_extrapolation_used"),&b)||b){
        set_failure(f,"manifest scope overclaims or differs from E2");return 0;
    }
    return 1;
}

static void json_escape(FILE *fp,const char *s) {
    const unsigned char *p=(const unsigned char*)s;fputc('"',fp);
    while(*p){if(*p=='"'||*p=='\\'){fputc('\\',fp);fputc(*p,fp);}else if(*p=='\n')fputs("\\n",fp);
        else if(*p<0x20)fprintf(fp,"\\u%04x",*p);else fputc(*p,fp);
        ++p;
    }
    fputc('"',fp);
}

static int write_report(const char *path,const Failure *f,const PairResult *pairs,size_t completed,
                        double wall,long peak_rss) {
    FILE *fp=fopen(path,"wb");size_t i;
    if(!fp)return 0;
    fputs("{\"contract_binding\":{\"parent_contract_sha256\":\"",fp);fputs(V05_SHA,fp);
    fputs("\",\"v0_6_contract_sha256\":\"",fp);fputs(V06_SHA,fp);
    fputs("\"},\"failure_reason\":",fp);if(f->failed)json_escape(fp,f->text);else fputs("null",fp);
    fprintf(fp,",\"pair_count_completed\":%zu,\"pairs\":[",completed);
    for(i=0;i<completed;++i){if(i)fputc(',',fp);fputs("{\"canonical_gray_byte_length\":589824,\"canonical_gray_full_bytes_equal\":true,\"canonical_png_full_bytes_equal\":true,\"cap\":",fp);
        json_escape(fp,pairs[i].cap_name);fputs(",\"graph_semantically_non_equivalent\":true,\"left_canonical_gray_sha256\":\"",fp);
        fputs(pairs[i].left_gray_sha,fp);fputs("\",\"left_canonical_png_sha256\":\"",fp);fputs(pairs[i].left_png_sha,fp);
        fputs("\",\"left_graph_sha256\":\"",fp);fputs(pairs[i].left_graph_sha,fp);fputs("\",\"left_padded_gray_sha256\":\"",fp);fputs(pairs[i].left_padded_sha,fp);
        fputs("\",\"left_raw_gray_sha256\":\"",fp);fputs(pairs[i].left_raw_sha,fp);fprintf(fp,"\",\"left_tuple_id\":%" PRId64 ",\"padded_clipping_certificates_passed\":true,\"pair_id\":",pairs[i].left_tuple_id);
        json_escape(fp,pairs[i].pair_id);fprintf(fp,",\"png_byte_length\":%zu,\"raw_gray_full_bytes_distinct\":true,\"right_canonical_gray_sha256\":\"",pairs[i].png_length);
        fputs(pairs[i].right_gray_sha,fp);fputs("\",\"right_canonical_png_sha256\":\"",fp);fputs(pairs[i].right_png_sha,fp);
        fputs("\",\"right_graph_sha256\":\"",fp);fputs(pairs[i].right_graph_sha,fp);fputs("\",\"right_padded_gray_sha256\":\"",fp);fputs(pairs[i].right_padded_sha,fp);
        fputs("\",\"right_raw_gray_sha256\":\"",fp);fputs(pairs[i].right_raw_sha,fp);fprintf(fp,"\",\"right_tuple_id\":%" PRId64 ",\"tuple_membership_recomputed\":true,\"width_px\":%d}",pairs[i].right_tuple_id,pairs[i].width);
    }
    fprintf(fp,"],\"peak_rss_kib_upper_bound\":%ld,\"runtime\":\"C11/GCC/zlib/4-process\",\"schema_version\":\"chemgraph-e2-verifier-b-report-v0.6\",\"status\":\"%s\",\"training_authorized\":false,\"wall_seconds_milli\":%.0f,\"workers\":4}\n",
            peak_rss,f->failed?"NO_GO":"passed",wall*1000.0);
    return fclose(fp)==0;
}

static int write_exact(int fd,const void *buffer,size_t length) {
    const unsigned char *p=buffer;
    while(length){ssize_t n=write(fd,p,length);if(n<0&&errno==EINTR)continue;if(n<=0)return 0;p+=n;length-=(size_t)n;}
    return 1;
}

static int read_exact(int fd,void *buffer,size_t length) {
    unsigned char *p=buffer;
    while(length){ssize_t n=read(fd,p,length);if(n<0&&errno==EINTR)continue;if(n<=0)return 0;p+=n;length-=(size_t)n;}
    return 1;
}

static void worker_run(unsigned worker,const Json *pairs,const char *root_dir,int output_fd) {
    WorkerReport report;unsigned index;
    memset(&report,0,sizeof(report));
    for(index=worker*2;index<worker*2+2;++index){
        if(!verify_pair(pairs->as.array.items[index],root_dir,report.seen,&report.pairs[report.count],&report.failure))break;
        report.artifact_bytes += 2ULL*(RAW_BYTES+PAD_BYTES+RAW_BYTES+report.pairs[report.count].png_length);
        ++report.count;
    }
    report.ok=!report.failure.failed&&report.count==2;
    if(!write_exact(output_fd,&report,sizeof(report)))_exit(3);
    close(output_fd);
    _exit(report.ok?0:1);
}

static void usage(const char *program) {
    fprintf(stderr,"usage: %s --manifest FILE --root DIR --contract-v06 FILE --contract-v05 FILE --output FILE\n",program);
}

int main(int argc,char **argv) {
    const char *manifest=NULL,*root_dir=NULL,*contract06=NULL,*contract05=NULL,*output=NULL,*schema,*status,*tree;
    Json *doc=NULL;const Json *pairs;Failure failure={{0},0};PairResult results[PAIR_COUNT];unsigned seen[2][4]={{0}};
    size_t completed=0,i,j;int64_t pair_count,artifact_count,artifact_total;uint64_t computed_total=0;
    struct timespec begin,end;struct rusage usage_info;char parse_error[1024]={0};long rss_upper=0;
    clock_gettime(CLOCK_MONOTONIC,&begin);memset(results,0,sizeof(results));
    for(i=1;i<(size_t)argc;++i){
        if(i+1>=(size_t)argc){usage(argv[0]);return 2;}
        if(!strcmp(argv[i],"--manifest"))manifest=argv[++i];else if(!strcmp(argv[i],"--root"))root_dir=argv[++i];
        else if(!strcmp(argv[i],"--contract-v06"))contract06=argv[++i];else if(!strcmp(argv[i],"--contract-v05"))contract05=argv[++i];
        else if(!strcmp(argv[i],"--output"))output=argv[++i];else{usage(argv[0]);return 2;}
    }
    if(!manifest||!root_dir||!contract06||!contract05||!output){usage(argv[0]);return 2;}
    if(!file_hash_is(contract06,V06_SHA,&failure)||!file_hash_is(contract05,V05_SHA,&failure))goto finish;
    doc=json_parse_file(manifest,parse_error,sizeof(parse_error));
    if(!doc){set_failure(&failure,"%s",parse_error);goto finish;}
    if(doc->kind!=JSON_OBJECT){set_failure(&failure,"manifest root is not an object");goto finish;}
    schema=json_string(json_get(doc,"schema_version"));status=json_string(json_get(doc,"status"));tree=json_string(json_get(doc,"artifact_tree_sha256"));
    if(!schema||strcmp(schema,"chemgraph-e2-producer-manifest-v0.6")||!status||strcmp(status,"passed")||
       !lowercase_hex64(tree)){set_failure(&failure,"manifest root identity or status mismatch");goto finish;}
    if(!require_binding(doc,&failure)||!validate_witness_plan(doc,&failure)||!validate_scope(doc,&failure))goto finish;
    if(!json_int(json_get(doc,"pair_count"),&pair_count)||pair_count!=PAIR_COUNT||
       !json_int(json_get(doc,"artifact_count"),&artifact_count)||artifact_count!=64||
       !json_int(json_get(doc,"artifact_total_bytes"),&artifact_total)||artifact_total<0||artifact_total>2147483648LL){
        set_failure(&failure,"manifest counts or artifact resource gate mismatch");goto finish;
    }
    pairs=json_get(doc,"pairs");if(!pairs||pairs->kind!=JSON_ARRAY||pairs->as.array.length!=PAIR_COUNT){set_failure(&failure,"pairs must contain exactly eight records");goto finish;}
    for(i=0;i<PAIR_COUNT;++i){
        const char *id=json_string(json_get(pairs->as.array.items[i],"pair_id"));
        for(j=0;j<i;++j){const char *previous=json_string(json_get(pairs->as.array.items[j],"pair_id"));
            if(id&&previous&&!strcmp(id,previous)){set_failure(&failure,"duplicate pair_id");goto finish;}}
    }
    {
        int pipes[4][2],launched=0;pid_t pids[4]={0};WorkerReport reports[4];
        memset(reports,0,sizeof(reports));
        for(i=0;i<4;++i){
            if(pipe(pipes[i])){set_failure(&failure,"cannot create worker pipe");break;}
            pids[i]=fork();
            if(pids[i]<0){close(pipes[i][0]);close(pipes[i][1]);set_failure(&failure,"cannot fork fixed E2 worker");break;}
            if(pids[i]==0){
                size_t k;close(pipes[i][0]);
                for(k=0;k<i;++k)close(pipes[k][0]);
                worker_run((unsigned)i,pairs,root_dir,pipes[i][1]);
            }
            close(pipes[i][1]);++launched;
        }
        for(i=0;i<(size_t)launched;++i){
            int status=0;struct rusage child_usage;
            if(!read_exact(pipes[i][0],&reports[i],sizeof(reports[i])))set_failure(&failure,"short worker report");
            close(pipes[i][0]);
            if(wait4(pids[i],&status,0,&child_usage)<0){set_failure(&failure,"cannot wait for E2 worker");continue;}
            rss_upper += child_usage.ru_maxrss;
            if(!WIFEXITED(status)||WEXITSTATUS(status)!=0||!reports[i].ok){
                if(reports[i].failure.failed)set_failure(&failure,"worker %zu: %s",i,reports[i].failure.text);
                else set_failure(&failure,"worker %zu exited abnormally",i);
            }
        }
        if(launched!=4)set_failure(&failure,"fixed four-worker schedule was not launched");
        if(failure.failed)goto finish;
        for(i=0;i<4;++i){
            for(j=0;j<reports[i].count;++j)results[completed++]=reports[i].pairs[j];
            computed_total+=reports[i].artifact_bytes;
            for(j=0;j<8;++j){unsigned value=reports[i].seen[j/4][j%4];
                if(value&&seen[j/4][j%4]){set_failure(&failure,"worker reports duplicate cap/width witness");goto finish;}
                seen[j/4][j%4]|=value;
            }
        }
    }
    for(i=0;i<2;++i)for(j=0;j<4;++j)if(!seen[i][j]){set_failure(&failure,"witness Cartesian product incomplete");goto finish;}
    if(computed_total!=(uint64_t)artifact_total){set_failure(&failure,"artifact_total_bytes differs from verified sidecars");goto finish;}
finish:
    json_free(doc);clock_gettime(CLOCK_MONOTONIC,&end);getrusage(RUSAGE_SELF,&usage_info);rss_upper+=usage_info.ru_maxrss;
    {
        double wall=(double)(end.tv_sec-begin.tv_sec)+(double)(end.tv_nsec-begin.tv_nsec)/1e9;
        if(wall>1800.0&&!failure.failed)set_failure(&failure,"E2 wall resource gate exceeded");
        if((uint64_t)rss_upper*1024ULL>8589934592ULL&&!failure.failed)set_failure(&failure,"E2 RSS resource gate exceeded");
        if(!write_report(output,&failure,results,completed,wall,rss_upper)){
            fprintf(stderr,"cannot write verifier report: %s\n",output);return 2;
        }
    }
    if(failure.failed){fprintf(stderr,"verifier B NO_GO: %s\n",failure.text);return 1;}
    return 0;
}
