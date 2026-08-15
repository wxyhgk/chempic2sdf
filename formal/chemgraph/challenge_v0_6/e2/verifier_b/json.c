#define _POSIX_C_SOURCE 200809L
#include "json.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const unsigned char *data;
    size_t length;
    size_t cursor;
    unsigned depth;
    char *error;
    size_t error_size;
} Parser;

static void fail(Parser *p, const char *message) {
    if (p->error[0] == '\0')
        snprintf(p->error, p->error_size, "json byte %zu: %s", p->cursor, message);
}

static Json *new_value(Parser *p, JsonKind kind) {
    Json *v = calloc(1, sizeof(*v));
    if (!v) { fail(p, "out of memory"); return NULL; }
    v->kind = kind;
    return v;
}

void json_free(Json *v) {
    size_t i;
    if (!v) return;
    if (v->kind == JSON_STRING) free(v->as.string.data);
    else if (v->kind == JSON_ARRAY) {
        for (i = 0; i < v->as.array.length; ++i) json_free(v->as.array.items[i]);
        free(v->as.array.items);
    } else if (v->kind == JSON_OBJECT) {
        for (i = 0; i < v->as.object.length; ++i) {
            free(v->as.object.keys[i]);
            json_free(v->as.object.values[i]);
        }
        free(v->as.object.keys);
        free(v->as.object.values);
    }
    free(v);
}

static void whitespace(Parser *p) {
    while (p->cursor < p->length &&
           (p->data[p->cursor] == ' ' || p->data[p->cursor] == '\n' ||
            p->data[p->cursor] == '\r' || p->data[p->cursor] == '\t'))
        ++p->cursor;
}

static int hex4(Parser *p, unsigned *value) {
    unsigned v = 0, digit;
    size_t i;
    if (p->length - p->cursor < 4) return 0;
    for (i = 0; i < 4; ++i) {
        unsigned char c = p->data[p->cursor++];
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        else return 0;
        v = v * 16 + digit;
    }
    *value = v;
    return 1;
}

static int append_utf8(char **buf, size_t *len, size_t *cap, unsigned cp) {
    unsigned char out[4]; size_t n, i;
    if (cp <= 0x7f) { out[0] = (unsigned char)cp; n = 1; }
    else if (cp <= 0x7ff) {
        out[0] = 0xc0 | (cp >> 6); out[1] = 0x80 | (cp & 63); n = 2;
    } else if (cp <= 0xffff) {
        out[0] = 0xe0 | (cp >> 12); out[1] = 0x80 | ((cp >> 6) & 63);
        out[2] = 0x80 | (cp & 63); n = 3;
    } else {
        out[0] = 0xf0 | (cp >> 18); out[1] = 0x80 | ((cp >> 12) & 63);
        out[2] = 0x80 | ((cp >> 6) & 63); out[3] = 0x80 | (cp & 63); n = 4;
    }
    if (*len + n + 1 > *cap) {
        size_t next = *cap ? *cap * 2 : 32;
        while (next < *len + n + 1) next *= 2;
        char *grown = realloc(*buf, next);
        if (!grown) return 0;
        *buf = grown; *cap = next;
    }
    for (i = 0; i < n; ++i) (*buf)[(*len)++] = (char)out[i];
    (*buf)[*len] = '\0';
    return 1;
}

static char *parse_string_raw(Parser *p, size_t *out_length) {
    char *buf = NULL; size_t len = 0, cap = 0;
    if (p->cursor >= p->length || p->data[p->cursor++] != '"') return NULL;
    while (p->cursor < p->length) {
        unsigned char c = p->data[p->cursor++];
        unsigned cp;
        if (c == '"') { *out_length = len; return buf ? buf : calloc(1, 1); }
        if (c < 0x20) { fail(p, "control character in string"); break; }
        if (c == '\\') {
            if (p->cursor >= p->length) { fail(p, "truncated escape"); break; }
            c = p->data[p->cursor++];
            if (c == '"' || c == '\\' || c == '/') cp = c;
            else if (c == 'b') cp = '\b';
            else if (c == 'f') cp = '\f';
            else if (c == 'n') cp = '\n';
            else if (c == 'r') cp = '\r';
            else if (c == 't') cp = '\t';
            else if (c == 'u') {
                if (!hex4(p, &cp)) { fail(p, "bad unicode escape"); break; }
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    unsigned low;
                    if (p->length - p->cursor < 6 || p->data[p->cursor] != '\\' ||
                        p->data[p->cursor + 1] != 'u') { fail(p, "missing low surrogate"); break; }
                    p->cursor += 2;
                    if (!hex4(p, &low) || low < 0xdc00 || low > 0xdfff) {
                        fail(p, "bad low surrogate"); break;
                    }
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                    fail(p, "unpaired low surrogate"); break;
                }
            } else { fail(p, "invalid escape"); break; }
        } else {
            if (c >= 0x80) { fail(p, "manifest must use escaped non-ASCII JSON"); break; }
            cp = c;
        }
        if (!append_utf8(&buf, &len, &cap, cp)) { fail(p, "out of memory"); break; }
    }
    free(buf);
    if (!p->error[0]) fail(p, "unterminated string");
    return NULL;
}

static Json *parse_value(Parser *p);

static Json *parse_array(Parser *p) {
    Json *v = new_value(p, JSON_ARRAY);
    if (!v) return NULL;
    ++p->cursor; whitespace(p);
    if (p->cursor < p->length && p->data[p->cursor] == ']') { ++p->cursor; return v; }
    while (!p->error[0]) {
        Json *item;
        Json **grown;
        whitespace(p); item = parse_value(p);
        if (!item) break;
        grown = realloc(v->as.array.items, (v->as.array.length + 1) * sizeof(*grown));
        if (!grown) { json_free(item); fail(p, "out of memory"); break; }
        v->as.array.items = grown;
        v->as.array.items[v->as.array.length++] = item;
        whitespace(p);
        if (p->cursor >= p->length) { fail(p, "unterminated array"); break; }
        if (p->data[p->cursor] == ']') { ++p->cursor; return v; }
        if (p->data[p->cursor++] != ',') { fail(p, "expected array comma"); break; }
    }
    json_free(v); return NULL;
}

static Json *parse_object(Parser *p) {
    Json *v = new_value(p, JSON_OBJECT);
    if (!v) return NULL;
    ++p->cursor; whitespace(p);
    if (p->cursor < p->length && p->data[p->cursor] == '}') { ++p->cursor; return v; }
    while (!p->error[0]) {
        char *key; size_t key_len, i; Json *item; char **keys; Json **values;
        whitespace(p);
        if (p->cursor >= p->length || p->data[p->cursor] != '"') {
            fail(p, "expected object key"); break;
        }
        key = parse_string_raw(p, &key_len);
        if (!key) break;
        for (i = 0; i < v->as.object.length; ++i) {
            if (!strcmp(key, v->as.object.keys[i])) {
                free(key); fail(p, "duplicate object key"); goto bad;
            }
        }
        whitespace(p);
        if (p->cursor >= p->length || p->data[p->cursor++] != ':') {
            free(key); fail(p, "expected object colon"); break;
        }
        whitespace(p); item = parse_value(p);
        if (!item) { free(key); break; }
        keys = realloc(v->as.object.keys, (v->as.object.length + 1) * sizeof(*keys));
        if (!keys) { free(key); json_free(item); fail(p, "out of memory"); break; }
        v->as.object.keys = keys;
        values = realloc(v->as.object.values, (v->as.object.length + 1) * sizeof(*values));
        if (!values) { free(key); json_free(item); fail(p, "out of memory"); break; }
        v->as.object.values = values;
        v->as.object.keys[v->as.object.length] = key;
        v->as.object.values[v->as.object.length++] = item;
        whitespace(p);
        if (p->cursor >= p->length) { fail(p, "unterminated object"); break; }
        if (p->data[p->cursor] == '}') { ++p->cursor; return v; }
        if (p->data[p->cursor++] != ',') { fail(p, "expected object comma"); break; }
    }
bad:
    json_free(v); return NULL;
}

static Json *parse_value(Parser *p) {
    Json *v; size_t start; char number[64], *end; long long parsed;
    if (++p->depth > 64) { fail(p, "nesting exceeds 64"); --p->depth; return NULL; }
    whitespace(p);
    if (p->cursor >= p->length) { fail(p, "missing value"); --p->depth; return NULL; }
    if (p->data[p->cursor] == '{') v = parse_object(p);
    else if (p->data[p->cursor] == '[') v = parse_array(p);
    else if (p->data[p->cursor] == '"') {
        size_t len; char *s = parse_string_raw(p, &len);
        v = s ? new_value(p, JSON_STRING) : NULL;
        if (v) { v->as.string.data = s; v->as.string.length = len; }
        else free(s);
    } else if (p->length - p->cursor >= 4 && !memcmp(p->data + p->cursor, "true", 4)) {
        p->cursor += 4; v = new_value(p, JSON_BOOL); if (v) v->as.boolean = 1;
    } else if (p->length - p->cursor >= 5 && !memcmp(p->data + p->cursor, "false", 5)) {
        p->cursor += 5; v = new_value(p, JSON_BOOL);
    } else if (p->length - p->cursor >= 4 && !memcmp(p->data + p->cursor, "null", 4)) {
        p->cursor += 4; v = new_value(p, JSON_NULL);
    } else {
        start = p->cursor;
        if (p->data[p->cursor] == '-') ++p->cursor;
        if (p->cursor >= p->length || !isdigit(p->data[p->cursor])) {
            fail(p, "invalid value"); --p->depth; return NULL;
        }
        if (p->data[p->cursor] == '0') ++p->cursor;
        else while (p->cursor < p->length && isdigit(p->data[p->cursor])) ++p->cursor;
        if (p->cursor < p->length && (p->data[p->cursor] == '.' ||
            p->data[p->cursor] == 'e' || p->data[p->cursor] == 'E')) {
            fail(p, "floating point is forbidden"); --p->depth; return NULL;
        }
        if (p->cursor - start >= sizeof(number)) {
            fail(p, "integer too long"); --p->depth; return NULL;
        }
        memcpy(number, p->data + start, p->cursor - start);
        number[p->cursor - start] = '\0'; errno = 0;
        parsed = strtoll(number, &end, 10);
        if (errno == ERANGE || *end) { fail(p, "integer outside int64"); --p->depth; return NULL; }
        v = new_value(p, JSON_INT); if (v) v->as.integer = (int64_t)parsed;
    }
    --p->depth; return v;
}

Json *json_parse_file(const char *path, char *error, size_t error_size) {
    FILE *f; long size; unsigned char *data; size_t got; Parser p; Json *root;
    if (error_size) error[0] = '\0';
    f = fopen(path, "rb");
    if (!f) { snprintf(error, error_size, "cannot open manifest: %s", strerror(errno)); return NULL; }
    if (fseek(f, 0, SEEK_END) || (size = ftell(f)) < 0 || size > 16 * 1024 * 1024L ||
        fseek(f, 0, SEEK_SET)) {
        fclose(f); snprintf(error, error_size, "manifest size invalid"); return NULL;
    }
    data = malloc((size_t)size + 1);
    if (!data) { fclose(f); snprintf(error, error_size, "out of memory"); return NULL; }
    got = fread(data, 1, (size_t)size, f);
    if (fclose(f) || got != (size_t)size) {
        free(data); snprintf(error, error_size, "cannot read manifest"); return NULL;
    }
    data[size] = 0;
    p = (Parser){data, (size_t)size, 0, 0, error, error_size};
    root = parse_value(&p); whitespace(&p);
    if (root && p.cursor != p.length) { fail(&p, "trailing JSON bytes"); json_free(root); root = NULL; }
    free(data); return root;
}

const Json *json_get(const Json *object, const char *key) {
    size_t i;
    if (!object || object->kind != JSON_OBJECT) return NULL;
    for (i = 0; i < object->as.object.length; ++i)
        if (!strcmp(object->as.object.keys[i], key)) return object->as.object.values[i];
    return NULL;
}

const char *json_string(const Json *v) {
    return v && v->kind == JSON_STRING ? v->as.string.data : NULL;
}

int json_int(const Json *v, int64_t *out) {
    if (!v || v->kind != JSON_INT) return 0;
    *out = v->as.integer; return 1;
}

int json_bool(const Json *v, int *out) {
    if (!v || v->kind != JSON_BOOL) return 0;
    *out = v->as.boolean; return 1;
}
