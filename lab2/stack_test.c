/*
 * stack_test.c
 *
 *  Created on: 18 Oct 2011
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
#if MEASURE == 0
#define DEBUG
#endif

#ifndef DEBUG
#define NDEBUG
#endif

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>
#include <stdbool.h>
#include <semaphore.h>

#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

stack_t *stack;
data_t data;
data_t poppedData;

void
test_init()
{
  // Initialize your test batch
}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  stack = stack_alloc();
  stack_init(stack, sizeof(data_t));

  data = DATA_VALUE;
  poppedData = 0;
}

void
test_teardown()
{
  // Do not forget to free your stacks after each test
  // to avoid memory leaks as now

  free(stack);
}

void
test_finalize()
{
  // Destroy properly your test batch
}


int
test_basic_stack()
{
  // Test that the stack works as expected on a single thread
  data_t data1 = 8;
  data_t data2 = 42;
  data_t data3 = 13;

  stack_push(stack, &data1);
  stack_push(stack, &data2);

  stack_pop(stack, &poppedData);
  assert(poppedData == 42);

  stack_push(stack, &data3);

  stack_pop(stack, &poppedData);
  assert(poppedData == 13);

  stack_pop(stack, &poppedData);
  assert(poppedData == 8);

  return 1;
}

struct thread_test_push_args
{
  int id;
};
typedef struct thread_test_push_args thread_test_push_args_t;


void*
thread_test_push(void* arg)
{
  int i;
  for (i = 0; i < MAX_PUSH_POP; i++) {
    stack_push(stack, &data);
  }

  return NULL;
}

int
test_push_safe()
{
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it

  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_push_args_t args[NB_THREADS];

  int i, success;

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      pthread_create(&thread[i], &attr, &thread_test_push, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  element_t *head = stack->head;
  int length = 0;

  while(head) {
    length++;
    head = head->next;
  }

  int expected = (size_t)(NB_THREADS * MAX_PUSH_POP);
  success = length == expected;

  if (!success)
  {
    printf("length is %d, expected %d...", length, expected);
  }

  return success;
}

void*
thread_test_pop(void* arg)
{
  int i;
  for (i = 0; i < MAX_PUSH_POP; i++) {
    stack_pop(stack, &poppedData);
    assert(poppedData == DATA_VALUE);
  }

  return NULL;
}

int
test_pop_safe()
{
  // Same as the test above for parallel pop operation
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_push_args_t args[NB_THREADS];

  int i, success;

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // push some elements for us to pop...
  for (i = 0; i < MAX_PUSH_POP * NB_THREADS; i++)
    {
      stack_push(stack, &data);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      pthread_create(&thread[i], &attr, &thread_test_pop, (void*) &args[i]);
    }
  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = stack->head == NULL;

  if (!success)
  {
    printf("head is at %p, expected NULL... ", &stack->head);
  }

  return success;

  return 0;
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

element_t *elemA, *elemB, *shared, *trash;
sem_t t0_semaphore, t1_semaphore, t2_semaphore;

int
stack_push_shared(stack_t *stack, element_t* newElem)
{
  element_t* old;

  do
  {
    old = stack->head;
    newElem->next = old;
  }
  while (cas((size_t *)&stack->head, (size_t)old, (size_t)newElem) != (size_t)old) ;
 
  return 0;
}


int
stack_pop_shared(stack_t *stack, element_t** buffer)
{
  element_t *old;

  do
  {
    old = stack->head;
  } while (cas((size_t *)&stack->head, (size_t)old, (size_t)old->next) != (size_t)old);

  *buffer = old;

  return 0;
}


int
stack_pop_unlucky(stack_t *stack, element_t** buffer)
{
  element_t *old;
  element_t *old_next = malloc(sizeof(element_t*));

  do
  {
    old = stack->head;
    memcpy(old_next, old->next, sizeof(element_t*));

    printf("old is %p, head is %p, next is %p\n", old, stack->head, old->next);

    // Tell T1 to start
    sem_post(&t1_semaphore);
    // Wait until it is time for T0 to go ahead and ruin everything
    sem_wait(&t0_semaphore);

    printf("old is %p, head is %p, next is %p\n", old, stack->head, old->next);
  } while (cas((size_t *)&stack->head, (size_t)old, (size_t)old_next) != (size_t)old) ;

  *buffer = old;
  
  return 0;
}

void*
thread_aba_t0(void* arg)
{
  printf("T0 pops A {\n");
    stack_pop_unlucky(stack, &trash);
  printf("} T0\n\n");

  return NULL;
}

void*
thread_aba_t1(void* arg)
{
  sem_wait(&t1_semaphore);

  printf(" T1 pops A {\n");
    stack_pop_shared(stack, &shared);
  printf(" } %p T1\n\n", shared);

  sem_post(&t2_semaphore);
  sem_wait(&t1_semaphore);

  printf(" T1 pushes A %p {\n", shared);
    stack_push_shared(stack, shared);
  printf(" } T1\n\n");

  sem_post(&t0_semaphore);
  return NULL;
}

void*
thread_aba_t2(void* arg)
{
  sem_wait(&t2_semaphore);

  printf(" T2 pops B {\n");
    stack_pop_shared(stack, &trash);
  printf(" } T2\n\n");

  sem_post(&t1_semaphore);

  return NULL;
}



int
test_aba()
{
  int success, aba_detected = 0;
  pthread_t t0, t1, t2;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  elemA = malloc(sizeof(element_t));
  elemB = malloc(sizeof(element_t));

  stack_push_shared(stack, elemB);
  stack_push_shared(stack, elemA);

  sem_init(&t0_semaphore, 0, 0);
  sem_init(&t1_semaphore, 0, 0);
  sem_init(&t2_semaphore, 0, 0);

  pthread_create(&t0, &attr, &thread_aba_t0, NULL);
  pthread_create(&t1, &attr, &thread_aba_t1, NULL);
  pthread_create(&t2, &attr, &thread_aba_t2, NULL);

  pthread_join(t0, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);

  aba_detected = (stack->head != NULL);

  success = aba_detected;
  return success;
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
      } while (cas(args->counter, old, local) != old);
    }

  return NULL;
}

int
test_cas()
{
#if 1
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = counter == (size_t)(NB_THREADS * MAX_PUSH_POP);

  if (!success)
    {
      printf("Got %ti, expected %i\n", counter, NB_THREADS * MAX_PUSH_POP);
    }

  assert(success);

  return success;
#else
  int a, b, c, *a_p, res;
  a = 1;
  b = 2;
  c = 3;

  a_p = &a;

  printf("&a=%X, a=%d, &b=%X, b=%d, &c=%X, c=%d, a_p=%X, *a_p=%d; cas returned %d\n", (unsigned int)&a, a, (unsigned int)&b, b, (unsigned int)&c, c, (unsigned int)a_p, *a_p, (unsigned int) res);

  res = cas((void**)&a_p, (void*)&c, (void*)&b);

  printf("&a=%X, a=%d, &b=%X, b=%d, &c=%X, c=%d, a_p=%X, *a_p=%d; cas returned %X\n", (unsigned int)&a, a, (unsigned int)&b, b, (unsigned int)&c, c, (unsigned int)a_p, *a_p, (unsigned int)res);

  return 0;
#endif
}

// Stack performance test
#if MEASURE != 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;
#endif

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  test_run(test_cas);

  test_run(test_basic_stack);
  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);

  test_finalize();
#else
  // Run performance tests
  int i, j;
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  stack_measure_arg_t arg[NB_THREADS];

  test_setup();

  // push some elements to have something to pop
  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      stack_push(stack, &data);
    }


#if MEASURE == 2
    for (j = 0; j < MAX_PUSH_POP; j++) {
      stack_push(stack, &data);
    }
#endif

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
      
      // Run push-based performance test based on MEASURE token
#if MEASURE == 1
      clock_gettime(CLOCK_MONOTONIC, &t_start[i]);

      // Push MAX_PUSH_POP times in parallel  
      pthread_create(&thread[i], &attr, &thread_test_push, (void*) &arg[i]);

      clock_gettime(CLOCK_MONOTONIC, &t_stop[i]);
#else

      // Run pop-based performance test based on MEASURE token
      clock_gettime(CLOCK_MONOTONIC, &t_start[i]);

      pthread_create(&thread[i], &attr, &thread_test_pop, (void*) &arg[i]);

      // Pop MAX_PUSH_POP times in parallel
      clock_gettime(CLOCK_MONOTONIC, &t_stop[i]);
#endif
    }

    //// BARRIER...

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  // Wait for all threads to finish
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
      printf("%i %i %li %i %li %i %li %i %li\n", i, (int) start.tv_sec,
          start.tv_nsec, (int) stop.tv_sec, stop.tv_nsec,
          (int) t_start[i].tv_sec, t_start[i].tv_nsec, (int) t_stop[i].tv_sec,
          t_stop[i].tv_nsec);
    }
#endif

  return 0;
}
