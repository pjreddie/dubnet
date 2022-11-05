OPENCV=0
OPENMP=0
DEBUG=0

OBJ=tensor.o matrix.o connected_layer.o activation_layer.o convolutional_layer.o maxpool_layer.o net.o data.o image.o classifier.o
EXOBJ=main.o test.o

VPATH=./src/:./:./lib/
EXEC=dubnet
SLIB=lib${EXEC}.so
ALIB=lib${EXEC}.a
OBJDIR=./obj/

CC=gcc
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread lib/jcr/libjcr.a
COMMON= -Iinclude/ -Isrc/ -Ilib/jcr/include/
CFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC 

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

EXOBJS = $(addprefix $(OBJDIR), $(EXOBJ))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile 

all: obj lib/jcr/libjcr.a $(SLIB) $(ALIB) $(EXEC)

lib/jcr/libjcr.a:
	cd lib/jcr && $(MAKE)

debug: OPTS = -O0 -g
debug: clean $(EXEC)

valgrind: debug
	valgrind --leak-check=full ./$(EXEC)

$(EXEC): $(EXOBJS) $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) 

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXOBJS) $(OBJDIR)/*

test: $(EXEC)
	./$(EXEC) test

