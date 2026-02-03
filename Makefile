CC := cc
CFLAGS := -Wall -Wextra -g
LDLIBS := -lm

PRINT ?= 0 # Can be 0 (no print) 1 (standard) 2 (verbose)
NUM_NEIGHBORS ?= 20

ENV_FLAGS := -DPRINT=${PRINT} -DNUM_NEIGHBORS=${NUM_NEIGHBORS}

# ---------------------------------------------------------
# OPERATING SYSTEM
# ---------------------------------------------------------
UNAME_S := $(shell uname -s)

# Default to Linux/Cluster flags (Standard GCC)
OMP_FLAGS := -fopenmp

# If macOS is detected, overwrite with Apple Clang flags
ifeq ($(UNAME_S),Darwin)
    # Force Homebrew path for Apple Silicon
    OMP_FLAGS := -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
endif

# ---------------------------------------------------------
# SOURCE FILES
# ---------------------------------------------------------

# List of files tat should become an executable
MAIN_SRCS := src/main.c src/omp_test.c src/scale_out.c

ALL_SRCS := $(wildcard src/*.c)

# Filter out mains
SHARED_SRCS := $(filter-out $(MAIN_SRCS), $(ALL_SRCS))

# ---------------------------------------------------------
# TARGETS
# ---------------------------------------------------------

.PHONY: all clean help

all: main

# Standard O0
main: src/main.c $(SHARED_SRCS)
	$(CC) $(CFLAGS) $(ENV_FLAGS) -g -fno-omit-frame-pointer -DNDEBUG -O0  $^ -o $@ $(LDLIBS)

# Standard optimized O3
main_opt: src/main.c $(SHARED_SRCS)
	$(CC) $(CFLAGS) $(ENV_FLAGS) -DNDEBUG -O3 $^ -o $@ $(LDLIBS)

# Multi-lm using OMP
scale-out: src/scale_out.c $(SHARED_SRCS)
	$(CC) $(CFLAGS) $(ENV_FLAGS) $(OMP_FLAGS) -g -fno-omit-frame-pointer -DNDEBUG -O3 $^ -o $@ $(LDLIBS)

# ---------------------------------------------------------
# GCC Build Rule (To fix the 256-thread limit on macOS from kmp)
# Requires: brew install gcc
# ---------------------------------------------------------
GCC_BIN ?= gcc-15

scale-out-gcc: src/scale_out.c $(SHARED_SRCS)
	$(GCC_BIN) $(CFLAGS) $(ENV_FLAGS) -fopenmp -g -fno-omit-frame-pointer -DNDEBUG -O3 $^ -o $@ $(LDLIBS)

clean:
	rm -f main main_opt scale-out scale-out-gcc *.o *~ 
	rm -rf *.dSYM

help:
	@echo "Detected OS: $(UNAME_S)"
	@echo "OMP Flags:   $(OMP_FLAGS)"
	@echo ""
	@echo "Targets:"
	@echo "  make scale-out   : Compiles src/scale_out.c for the current OS"
	@echo "  make scale-out-gcc   : Compiles src/scale_out.c for the current OS using gcc explicitly"
