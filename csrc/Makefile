# Standalone Makefile for Lila ASI v2.6 C11 Engine (TicketLabs Open-Source Bundle)
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)
CC := gcc
CFLAGS := -O3 -Wall -Wextra -std=c11 -pthread -fopenmp -D_GNU_SOURCE
LDFLAGS := -lm -lpthread -fopenmp

ifeq ($(UNAME_S),Linux)
    LDFLAGS += -Wl,-z,noexecstack
endif

ifeq ($(UNAME_M),x86_64)
    CFLAGS += -march=native -mavx2 -mfma -mf16c
else ifneq ($(filter $(UNAME_M),aarch64 arm64),)
    CFLAGS += -march=native
endif

LIB_SRC := asi_ed25519.c asi_v2.c asi_arch.c asi_dense.c lila_lm_head_candidate.c lila_asi.c asi_grammar_sparse.c e05_bridge.c asi_state_centric.c asi_kernels.c asi_graph.c asi_pairing.c asi_vision.c

.PHONY: all clean

all: ../lib/liblila_asi.so ../lib/libasi_v2.so ../bin/lila-asi-cli

../lib/liblila_asi.so: $(LIB_SRC)
	mkdir -p ../lib ../python/lila_asi
	$(CC) $(CFLAGS) -fPIC -shared -I . -o $@ $(LIB_SRC) $(LDFLAGS)
	cp -f $@ ../python/lila_asi/liblila_asi.so

../lib/libasi_v2.so: asi_v2.c asi_ed25519.c
	mkdir -p ../lib ../python/lila_asi
	$(CC) $(CFLAGS) -fPIC -shared -I . -o $@ asi_v2.c asi_ed25519.c
	cp -f $@ ../python/lila_asi/libasi_v2.so

../bin/lila-asi-cli: asi_v2_cli.c $(LIB_SRC)
	mkdir -p ../bin
	$(CC) $(CFLAGS) -I . asi_v2_cli.c $(LIB_SRC) -o $@ $(LDFLAGS)

clean:
	rm -f ../lib/liblila_asi.so ../lib/libasi_v2.so ../bin/lila-asi-cli ../python/lila_asi/*.so
