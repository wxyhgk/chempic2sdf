CC ?= gcc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
LDLIBS ?= -lz

TARGET := verify_e2_v0_6_b
SOURCES := verify_e2_v0_6_b.c json.c sha256.c
HEADERS := json.h sha256.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) $(LDLIBS) -o $@

clean:
	rm -f $(TARGET)
