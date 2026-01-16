#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <popt.h>

#include "life.h"
#include "load.h"
#include "save.h"

static void
copy_board(char outboard[], const char inboard[], const int nrows,
	   const int ncols)
{
	/* We use memmove in case outboard and inboard overlap (in this
	   case, this would mean that they are the same */
	memmove(outboard, inboard, nrows * ncols * sizeof(char));
}

static int
boards_equalp(const char b1[], const char b2[], const int nrows,
	      const int ncols)
{
	int i;
	for (i = 0; i < nrows * ncols; i++)
		if (b1[i] != b2[i])
			return 0;

	return 1;
}

/* program options */
int verify = 0;
int sequential = 0;

/* global variables, initialized in parse_args */
int num_generations = 0;
FILE *inputfile = NULL;
FILE *outputfile = NULL;

static void
parse_args(int argc, char *argv[]) {
	poptContext context; /* The context for popt parsing */
	int option;

    	struct poptOption options[] = {
		/* Long name, short name, argument type, address of storage,
		   value, description, argument description */
		{ "verify", 'v', POPT_ARG_NONE, &verify, 0,
		  "verify output", NULL },
		{ "sequential", 's', POPT_ARG_NONE, &sequential, 0,
		  "run original sequential code", NULL },
		POPT_AUTOHELP
		POPT_TABLEEND
	};

	context = poptGetContext("gol", argc, (const char **)argv, options, 0);
	poptSetOtherOptionHelp(context, "num_generations inputfile outputfile\n"
			       "       num_generations: number of generations\n"
			       "       inputfile:       input board file\n"
			       "       outputfile:      output board file "
			       "(use \"-\" for stdout)\n");

	while ((option = poptGetNextOpt(context)) >= 0);

	if (option < -1) {
		fprintf(stderr, "%s: %s\n", poptBadOption(context, 0),
			poptStrerror(option));
		poptPrintHelp(context, stderr, 0);
		exit(EXIT_FAILURE);
	}

	/* non-optional arguments */
	const char **args = poptGetArgs(context);

	if (args == NULL || args[0] == NULL || args[1] == NULL ||
	    args[2] == NULL) {
		fprintf(stderr, "error: need three arguments\n");
		poptPrintHelp(context, stderr, 0);
		exit(EXIT_FAILURE);
	}

	char *endptr;
	num_generations = strtol(args[0], &endptr, 10);
	if (num_generations < 0 || endptr[0] != 0) {
		fprintf(stderr, "num_generations: %s: must be >= 0\n",
			args[0]);
		poptPrintHelp(context, stderr, 0);
		exit(EXIT_FAILURE);
	}

	/* open input file */
	inputfile = fopen(args[1], "r");
	if (inputfile == NULL) {
		fprintf(stderr, "failed to open input file: %s\n", args[1]);
		poptPrintHelp(context, stderr, 0);
		exit(EXIT_FAILURE);
	}

	/* open output file, if output file is "-", use stdout */
	if (strcmp(args[2], "-") == 0) {
		outputfile = stdout;
	} else {
		outputfile = fopen(args[2], "w");
	}
	if (outputfile == NULL) {
		fprintf(stderr, "failed to open output file: %s\n", args[2]);
		poptPrintHelp(context, stderr, 0);
		exit(EXIT_FAILURE);
	}

	/* Free the popt context */
	poptFreeContext(context);
}

int
main(int argc, char *argv[])
{
	char *inboard = NULL;
	char *outboard = NULL;
	char *checkboard = NULL;
	char *final_board = NULL;
	int nrows = 0;
	int ncols = 0;

	parse_args(argc, argv);
	/* load initial board state from input file */
	inboard = load_board(inputfile, &nrows, &ncols);
	fclose(inputfile);

	/* create a second board for ping-ponging */
	outboard = make_board(nrows, ncols);

	if (verify) {
		/* make a third board, copy initial state into it */
		checkboard = make_board(nrows, ncols);
		copy_board(checkboard, inboard, nrows, ncols);
	}

	if (sequential) {
		/* run the original code */
		final_board = sequential_game_of_life(outboard, inboard, nrows,
						      ncols, num_generations);
	} else {
		/* you need to parallelize game_of_life() for this lab */
		final_board = game_of_life(outboard, inboard, nrows,
					   ncols, num_generations);
	}

	/* save the final board to output file */
	save_board(outputfile, final_board, nrows, ncols);
	if (outputfile != stdout) {
		fclose(outputfile);
	}

	if (verify) {
		/* we ping-pong between inboard and outboard, so final_board can
		 * be inboard or outboard. copy final_board to outboard so we
		 * can verify outboard. */
		copy_board(outboard, final_board, nrows, ncols);

		/* use checkboard as initial board */
		final_board =
			sequential_game_of_life(inboard, checkboard, nrows,
						ncols, num_generations);

		/* final_board contains correct ouput. check outboard */
		if (boards_equalp(final_board, outboard, nrows, ncols))
			printf("Verification successful\n");
		else {
			fprintf(stderr, "Verification failed\n");
			exit(EXIT_FAILURE);
		}
	}

	/* clean up */
	if (inboard != NULL)
		free(inboard);
	if (outboard != NULL)
		free(outboard);
	if (checkboard != NULL)
		free(checkboard);

	return 0;
}
