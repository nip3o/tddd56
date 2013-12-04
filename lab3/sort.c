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
 #include <pthread.h>

#ifndef DEBUG
#define NDEBUG
#endif

#include "array.h"
#include "sort.h"
#include "simple_quicksort.h"

int sequentialMergesort(value*, int);
void parallelMergesort(value*, int);
void sequentialMerge(value*, int, int);
void *parallel_thread(void*);

pthread_mutex_t mutex;
int thread_count;
int ARRAY_SIZE;

int
sort(struct array * array)
{
    printf("Begin sort\n");
    if (NB_THREADS == 0) {
        simple_quicksort_ascending(array);
    } else if (NB_THREADS == 1) {
        sequentialMergesort(array->data, array->length);
    } else {
        parallelMergesort(array->data, array->length);
    }

    return 0;
}

struct sorting_args {
    value* array;
    int length;
};

void *
run_thread(void* arg) {
    struct sorting_args * args = (struct sorting_args*)arg;

    sequentialMergesort(args->array, args->length);

    pthread_exit(NULL);
}

void parallelMergesort(value *array, int n) {
    pthread_mutex_init(&mutex, NULL);
    thread_count = 1;

    struct sorting_args arg;
    arg.array = array;
    arg.length = n;
    ARRAY_SIZE = n;

    parallel_thread(&arg);
}

void *
parallel_thread(void* arg) {
    struct sorting_args * args = (struct sorting_args*)arg;
    if (args->length == 1)
        return 0;

    int create = 0;
    if (args->length >= ARRAY_SIZE / 2) {
        /* Critical Section */
        pthread_mutex_lock(&mutex);
        if (thread_count < NB_THREADS) {
            thread_count++;
            create = 1;
        }
        pthread_mutex_unlock(&mutex);
    }

    int middle = args->length / 2;

    if (create == 1){
        struct sorting_args t_arg, t_arg2;
        pthread_t t;

        t_arg.array = args->array + middle;
        t_arg.length = args->length - middle;

        t_arg2.array = args->array;
        t_arg2.length = middle;

        pthread_create(&t, NULL, &parallel_thread, &t_arg);
        printf("Creating thread %d\n", t_arg.length);
        parallel_thread(&t_arg2);

        pthread_join(t, NULL);

    } else {
        sequentialMergesort(args->array, args->length);
    }

    sequentialMerge(args->array, middle, args->length);
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
