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
  unsigned int i;
  unsigned int val;

  const int group_size = get_local_size(0);
  const int local_id = get_local_id(0);
  const int id = get_global_id(0);
  const int group_id = get_group_id(0);

  int buffer_start = group_id * group_size;
  int buffer_stop = buffer_start + 512;

  __local unsigned int buffer[BUFFSIZE];

  val = data[id];
  buffer[id % 512] = val;


  barrier(CLK_LOCAL_MEM_FENCE);

  //find out how many values are smaller

    for (i = 0; i < get_global_size(0); i++) {
    	if ( buffer_start <= i && i < buffer_stop ) {
    		if (val > buffer[i % 512]) {
    			pos++;
    		}
      } else if (val > data[i]) {
        pos++;
      }
    }

  out[pos] = val;
}
