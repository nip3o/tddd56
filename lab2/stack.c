/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through lock-based CAS
#else
#warning Stacks are synchronized through hardware CAS
#endif
#endif

stack_t *
stack_alloc()
{
  // Example of a task allocation with correctness control
  // Feel free to change it
  stack_t *res;

  res = malloc(sizeof(stack_t));
  assert(res != NULL);

  if (res == NULL)
    return NULL;

// You may allocate a lock-based or CAS based stack in
// different manners if you need so
#if NON_BLOCKING == 0
  // Implement a lock_based stack
#elif NON_BLOCKING == 1
  /*** Optional ***/
  // Implement a software CAS-based stack
#else
  // Implement a hardware CAS-based stack
#endif

  return res;
}

int
stack_init(stack_t *stack, size_t size)
{
  assert(stack != NULL);
  assert(size > 0);

  stack->head = NULL;
  stack->elementSize = size;


#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_init(&stack->lock, NULL);

#elif NON_BLOCKING == 1
  /*** Optional ***/
  // Implement a software CAS-based stack
#else
  // Implement a hardware CAS-based stack
#endif

  return 0;
}

int
stack_check(stack_t *stack)
{
  /*** Optional ***/
  // Use code and assertions to make sure your stack is
  // in a consistent state and is safe to use.
  //
  // For now, just makes just the pointer is not NULL
  //
  // Debugging use only

  assert(stack != NULL);

  return 0;
}

int
stack_push(stack_t *stack, void* buffer)
{

#if NON_BLOCKING == 0

  // Implement a lock_based stack

  pthread_mutex_lock(&stack->lock);
    element_t *newElem = malloc(sizeof(element_t));
    newElem->next = stack->head;
    newElem->data = malloc(stack->elementSize);
    memcpy(newElem->data, buffer, stack->elementSize);

    stack->head = newElem;
  pthread_mutex_unlock(&stack->lock);

#elif NON_BLOCKING == 1
  /*** Optional ***/
  // Implement a software CAS-based stack
#else

  // Implement a hardware CAS-based stack

  element_t *newElem = malloc(sizeof(element_t));
  element_t* old;

  newElem->data = malloc(stack->elementSize);
  memcpy(newElem->data, buffer, stack->elementSize);

  do
  {
    old = stack->head;
    newElem->next = old;
  }
  while (cas((size_t *)&stack->head, (size_t)old, (size_t)newElem) != (size_t)old) ;
  
#endif

  return 0;
}

int
stack_pop(stack_t *stack, void* buffer)
{

#if NON_BLOCKING == 0
  // Implement a lock_based stack

  pthread_mutex_lock(&stack->lock);
    element_t *elem = stack->head;
    memcpy(buffer, elem->data, stack->elementSize);

    stack->head = elem->next;

    free(elem->data);
    free(elem);
  pthread_mutex_unlock(&stack->lock);


#elif NON_BLOCKING == 1
  /*** Optional ***/
  // Implement a software CAS-based stack
#else

  // Implement a hardware CAS-based stack

  element_t *old, *elem;
  do
  {
    elem = stack->head;
    memcpy(buffer, elem->data, stack->elementSize);

    old = stack->head;
  } while (cas((size_t *)&stack->head, (size_t)old, (size_t)elem->next) != (size_t)old) ;

  stack->head = elem->next;

#endif

  return 0;
}
