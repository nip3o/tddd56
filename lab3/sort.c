/*
 * sort.c
 *
 *  Created on: 5 Sep 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

// Do not touch or move these lines
#include <stdio.h>
#include "disable.h"
#include <stdlib.h>

#ifndef DEBUG
#define NDEBUG
#endif

#include "array.h"
#include "sort.h"
#include "simple_quicksort.h"

int sequentialMergesort(value*, int);
void sequentialMerge(value*, int, int);

int
sort(struct array * array)
{
	//simple_quicksort_ascending(array);

	sequentialMergesort(array->data, array->length);
	return 0;
}


int sequentialMergesort(value *array, int n) {
	if(n==1) {
		return 0;
	}

	int middle = n/2;
	sequentialMergesort(array, middle);
	sequentialMergesort(array + middle, n - middle);

	sequentialMerge(array, middle, n);
	return 0;
}

void sequentialMerge(value *array, int middle, int n) {
	int i0 = 0, i1 = middle, j;
	int* temp = malloc(n*sizeof(value));

	for (j = 0; j < n; j++) {
		if (i0 < middle && (i1 >= n || array[i0] <= array[i1])) {
			temp[j] = array[i0++];
		} else {
			temp[j] = array[i1++];
		}
	}
	for (j = 0; j < n; j++) {
		array[j] = temp[j];
	}
}
