#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "random_bit.h"

static void
print_usage(const char *cmd)
{
	fprintf(stderr, "Usage: %s nrows ncolumns\n", cmd);
}

int
main(int argc, char *argv[])
{
	int nrows, ncols;
	char *endptr;

	if (argc != 3) {
		print_usage(argv[0]);
		exit(EXIT_FAILURE);
	}

	nrows = strtol(argv[1], &endptr, 10);
	if (nrows < 0 || endptr[0] != 0) {
		fprintf(stderr, "nrows: %s: not an integer\n", argv[1]);
		print_usage(argv[0]);
		exit(EXIT_FAILURE);
	} else if (nrows < 1) {
		fprintf(stderr, "nrows: %d: should be > 0\n", nrows);
		print_usage(argv[0]);
		exit(EXIT_FAILURE);
	}

	ncols = strtol(argv[2], &endptr, 10);
	if (ncols < 0 || endptr[0] != 0) {
		fprintf(stderr, "ncols: %s: not an integer!\n", argv[2]);
		print_usage(argv[0]);
		exit(EXIT_FAILURE);
	} else if (ncols < 1) {
		fprintf(stderr, "ncols: %d: should be > 0\n", ncols);
		print_usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	
	init_random_bit();

	printf("P1\n%d %d\n", nrows, ncols);
	for (int ii = 0; ii < nrows; ii++) {
		for (int jj = 0; jj < ncols; jj++) {
			printf("%c\n", '0' + random_bit());
		}
	}
	return 0;
}
