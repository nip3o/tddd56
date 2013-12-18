/*
 * Rank sorting in sorting OpenCL
 * This kernel has a bug. What?
 */

#define BUFFSIZE 512

__kernel void sort(__global unsigned int *data, 
				   const unsigned int length, 
				   __global unsigned int *out)
{ 
  unsigned int pos = 0;
  unsigned int i, k;
  unsigned int val;

  const int group_size = get_local_size(0);
  const int local_id = get_local_id(0);
  const int id = get_global_id(0);
  const int group_id = get_group_id(0);

  int buffer_start = group_id * group_size;
  int buffer_stop = buffer_start + 512;

  val = data[id];

  __local unsigned int buffer[BUFFSIZE];

  //find out how many values are smaller
  for(k = 0; k < get_num_groups(0); k++) {
    barrier(CLK_LOCAL_MEM_FENCE);

    buffer_start = k * group_size;
    int offset = id % 512;
    buffer[offset] = data[buffer_start + offset];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for (i = 0; i < 512; i++) {
  		if (val > buffer[i]) {
  			pos++;
  		}
    }

  }

  out[pos] = val;
}
