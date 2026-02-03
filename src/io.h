#ifndef HELPER_H
#define HELPER_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define FREAD(ptr, size, num, fp) \
    do { \
        size_t read = fread(ptr, size, num, fp); \
        assert(read > 0); \
        (void) read; \
    } while(0)

#define FREAD_CHECK(ptr, size, num, fp) \
    do { \
        size_t read = fread(ptr, size, num, fp); \
        assert(read == num); \
        (void) read; \
    } while(0) 

#define FWRITE(ptr, size, num, fp) \
    do { \
        size_t written = fwrite(ptr, size, num, fp); \
        assert(written > 0); \
        (void) written; \
    } while(0)

#define FWRITE_CHECK(ptr, size, num, fp) \
    do { \
        size_t written = fwrite(ptr, size, num, fp); \
        assert(written == num); \
        (void) written; \
    } while(0)

#endif // HELPER_H
