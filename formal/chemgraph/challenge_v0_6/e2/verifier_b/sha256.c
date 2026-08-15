#include "sha256.h"
#include <string.h>

static const uint32_t round_constant[64] = {
  0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
  0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
  0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
  0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
  0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
  0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
  0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
  0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
static uint32_t load_be32(const unsigned char *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}
static void store_be32(unsigned char *p, uint32_t x) {
    p[0]=(unsigned char)(x>>24); p[1]=(unsigned char)(x>>16); p[2]=(unsigned char)(x>>8); p[3]=(unsigned char)x;
}

static void compress(Sha256 *c, const unsigned char block[64]) {
    uint32_t w[64], a,b,d,e,f,g,h,t1,t2,cc; unsigned i;
    for (i=0;i<16;++i) w[i]=load_be32(block+4*i);
    for (;i<64;++i) {
        uint32_t x=w[i-15], y=w[i-2];
        w[i]=(rotr(y,17)^rotr(y,19)^(y>>10))+w[i-7]+(rotr(x,7)^rotr(x,18)^(x>>3))+w[i-16];
    }
    a=c->state[0]; b=c->state[1]; cc=c->state[2]; d=c->state[3];
    e=c->state[4]; f=c->state[5]; g=c->state[6]; h=c->state[7];
    for(i=0;i<64;++i){
        t1=h+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^((~e)&g))+round_constant[i]+w[i];
        t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&cc)^(b&cc));
        h=g; g=f; f=e; e=d+t1; d=cc; cc=b; b=a; a=t1+t2;
    }
    c->state[0]+=a; c->state[1]+=b; c->state[2]+=cc; c->state[3]+=d;
    c->state[4]+=e; c->state[5]+=f; c->state[6]+=g; c->state[7]+=h;
}

void sha256_init(Sha256 *c) {
    static const uint32_t initial[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    memcpy(c->state,initial,sizeof(initial)); c->byte_count=0; c->used=0;
}

void sha256_update(Sha256 *c, const void *payload, size_t length) {
    const unsigned char *p=payload; c->byte_count += length;
    while(length){
        size_t room=64-c->used, take=length<room?length:room;
        memcpy(c->block+c->used,p,take); c->used+=take; p+=take; length-=take;
        if(c->used==64){compress(c,c->block);c->used=0;}
    }
}

void sha256_final(Sha256 *c, unsigned char digest[32]) {
    uint64_t bits=c->byte_count*8; unsigned i;
    c->block[c->used++]=0x80;
    if(c->used>56){memset(c->block+c->used,0,64-c->used);compress(c,c->block);c->used=0;}
    memset(c->block+c->used,0,56-c->used);
    for(i=0;i<8;++i)c->block[63-i]=(unsigned char)(bits>>(8*i));
    compress(c,c->block);
    for(i=0;i<8;++i)store_be32(digest+4*i,c->state[i]);
    memset(c,0,sizeof(*c));
}

void sha256_hex(const void *data, size_t length, char hex[65]) {
    static const char alphabet[]="0123456789abcdef"; unsigned char digest[32]; unsigned i; Sha256 c;
    sha256_init(&c); sha256_update(&c,data,length); sha256_final(&c,digest);
    for(i=0;i<32;++i){hex[2*i]=alphabet[digest[i]>>4];hex[2*i+1]=alphabet[digest[i]&15];}
    hex[64]='\0';
}
