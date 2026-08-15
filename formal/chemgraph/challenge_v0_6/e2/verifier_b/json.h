#ifndef E2B_JSON_H
#define E2B_JSON_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    JSON_NULL, JSON_BOOL, JSON_INT, JSON_STRING, JSON_ARRAY, JSON_OBJECT
} JsonKind;

typedef struct Json Json;

struct Json {
    JsonKind kind;
    union {
        int boolean;
        int64_t integer;
        struct { char *data; size_t length; } string;
        struct { Json **items; size_t length; } array;
        struct { char **keys; Json **values; size_t length; } object;
    } as;
};

Json *json_parse_file(const char *path, char *error, size_t error_size);
void json_free(Json *value);
const Json *json_get(const Json *object, const char *key);
const char *json_string(const Json *value);
int json_int(const Json *value, int64_t *out);
int json_bool(const Json *value, int *out);

#endif
