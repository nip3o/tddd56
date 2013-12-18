/*
 * Placeholder for wavelet transform.
 * Currently just a simple invert.
 */

__kernel void kernelmain(__global unsigned char *image, __global unsigned char *data, const unsigned int length, unsigned int row_width)
{
	// data[get_global_id(0)] = image[get_global_id(0)];
	unsigned int id = get_global_id(0);
	unsigned char in1, in2, in3, in4, out;
	
	if (id < (length / 2)) {
		if ((id % row_width) < (row_width / 2)) {
			unsigned int id2;
			if (id % 3 == 0)
				id2 = id * 2;
			if (id % 3 == 1)
				id2 = (id - 1) * 2 + 1;
			if (id % 3 == 2)
				id2 = id * 2 + 1;

			in1 = image[id2],
	  		in2 = image[id2 + 3],
	  		in3 = image[id2 + row_width],
	  		in4 = image[id2 + row_width + 3];
			out = (in1 + in2 + in3 + in4)/4;
		} else {
			unsigned int id2;
			if (id % 3 == 0)
				id2 = (id - row_width / 2) * 2;
			if (id % 3 == 1)
				id2 = ((id - row_width / 2) - 1) * 2 + 1;
			if (id % 3 == 2)
				id2 = (id - row_width / 2) * 2 + 1;

			in1 = image[id2],
	  		in2 = image[id2 + 3],
	  		in3 = image[id2 + row_width],
	  		in4 = image[id2 + row_width + 3];
			out = (in1 + in2 - in3 - in4)/4 + 128;
		}
	} else {
		if ((id % row_width) < (row_width / 2)) {
			unsigned int id2;
			if (id % 3 == 0)
				id2 = (id - length / 2) * 2;
			if (id % 3 == 1)
				id2 = ((id - length / 2) - 1) * 2 + 1;
			if (id % 3 == 2)
				id2 = (id - length / 2) * 2 + 1;

			in1 = image[id2],
	  		in2 = image[id2 + 3],
	  		in3 = image[id2 + row_width],
	  		in4 = image[id2 + row_width + 3];
			out = (in1 - in2 + in3 - in4)/4 + 128;
		} else {
			unsigned int id2;
			if (id % 3 == 0)
				id2 = (id - length / 2 - row_width / 2) * 2;
			if (id % 3 == 1)
				id2 = ((id - length / 2 - row_width / 2) - 1) * 2 + 1;
			if (id % 3 == 2)
				id2 = (id - length / 2 - row_width / 2) * 2 + 1;

			in1 = image[id2],
	  		in2 = image[id2 + 3],
	  		in3 = image[id2 + row_width],
	  		in4 = image[id2 + row_width + 3];
			out = (in1 - in2 - in3 + in4)/4 + 128;
		}
	}
	data[id] = out;
}
