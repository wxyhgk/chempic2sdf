#ifndef E2B_SHA256_H
#define E2B_SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state[8];
    uint64_t byte_count;
    unsigned char block[64];
    size_t used;
} Sha256;

void sha256_init(Sha256 *ctx);
void sha256_update(Sha256 *ctx, const void *data, size_t length);
void sha256_final(Sha256 *ctx, unsigned char digest[32]);
void sha256_hex(const void *data, size_t length, char hex[65]);

#endif
