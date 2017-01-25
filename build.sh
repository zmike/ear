gcc -Wall -W -Wpointer-arith -Wshadow -Wno-missing-field-initializers \
    -Wfloat-equal -Wuninitialized -Wundef -Wcast-align -Wformat=2 -Wno-format-y2k \
    -o ear ear.c $(pkg-config --cflags --libs elementary emotion evas opencv) -g -O0
