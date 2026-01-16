#include "random_bit.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


static uint32_t
get_random_seed()
{
	FILE *r = fopen("/dev/urandom", "r");
	if (r == NULL) {
		fprintf(stderr, "Warning: failed to open /dev/urandom, "
			"using system clock for random seed\n");
		return (uint32_t)time(NULL);
	} 
	uint32_t seed;
	assert(fread(&seed, sizeof(uint32_t), 1, r) == 1);
	fclose(r);
	return seed;
}

void
init_random_bit()
{
	srand(get_random_seed());
}

char
random_bit()
{
	/* Hopefully this makes somewhat random bits. Accuracy / quality aren't
	 * so important for this application; we just want some initial data
	 * that isn't strongly patterned. */
	const int r = rand() / (RAND_MAX / 2);
	return (char)r;
}
