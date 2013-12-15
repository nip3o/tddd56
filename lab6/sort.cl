/*
 * Rank sorting in sorting OpenCL
 * This kernel has a bug. What?
 */

__kernel void sort(__global unsigned int *data, 
				   const unsigned int length, 
				   __global unsigned int *out)
{ 
  unsigned int pos = 0;
  unsigned int i;
  unsigned int val;

  //find out how many values are smaller
  for (i = 0; i < get_global_size(0); i++)
    if (data[get_global_id(0)] > data[i])
      pos++;

  val = data[get_global_id(0)];
  out[pos]=val;
}
