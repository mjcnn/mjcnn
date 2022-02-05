/* ------------------------------------------------------------------------- */
/*   Copyright (C) 2021 Marius C. Silaghi
                Author: Marius Silaghi: msilaghi@fit.edu
                Florida Tech, Human Decision Support Systems Laboratory
   
       This program is free software; you can redistribute it and/or modify
       it under the terms of the GNU Affero General Public License as published by
       the Free Software Foundation; either the current version of the License, or
       (at your option) any later version.
   
      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.
  
      You should have received a copy of the GNU Affero General Public License
      along with this program; if not, write to the Free Software
      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.              */
/* ------------------------------------------------------------------------- */
package cnn;

import cnn.util.Array1DI;
import cnn.util.Array1DIfromD3;
import cnn.util.Array2DF;
import cnn.util.Array3DF;
import cnn.util.Array3DI;

public
class LayerCPU implements LayerImplementation {
	Array2DF input;
	Array2DF output;
	
	final float softmax_aggregate[] = new float[1]; // used in softmax GPU kernels, for final aggregation result
	float softmax_aggregates[];                      // Used in softmax hierarchical aggregation
	
	Perceptron p[][];                                 // An array of perceptrons, by output depth and raster
	int X_input, Y_input, D_input, X_output, Y_output, D_output, type;

	/* Filter Arrays allocated outside, for efficient GPU access */
	public Array2DF bias = null;     // [output_depth.1]
	public Array3DF weights;       // [output_depth.input_depth.idx_raster_activation_field]  the raster is based on class Field
	public Array3DI indexes;          // [output_depth|1.raster_output.idx_raster_activation_field]
	
	public LayerCPU(
			Array2DF input, int X_i, int Y_i, int D_i,
			Array2DF output, int X_o, int Y_o, int D_o,
			int x_pad, int y_pad, int stride, int dilation,
			Filter filter[], Array2DF bias, Array3DF weights, int type, String name
			) {
		
		init(
				input,  X_i,  Y_i,  D_i,
				 output,  X_o,  Y_o,  D_o,
				 x_pad,  y_pad,  stride,  dilation,
				 filter,  bias,  weights,  type, name);
		
	}
	
	void initPerceptron(int output_layer, int x, int X_o, int y, int Y_o, int D_i, int xi, int X_i, int yi, int Y_i, int dilation) {
		Array1DI perceptron_indexes = null;
		int crt_index_layer = -1;
		boolean flat_input_depth = false;
		switch(type) {
		// different per output depth
		case Config.POOLMAX:
		case Config.POOLAVG: 
			if (Config.FLAT_INDEXES_POOL) crt_index_layer=0; else crt_index_layer = output_layer;
			perceptron_indexes = new Array1DIfromD3(indexes, crt_index_layer, Field.getIndex(x, X_o, y, Y_o)); break;
		// shared across output depth
		case Config.CONVOLUTION:
		case Config.RELU:
		case Config.SIGMOID:
		case Config.TANH: 
			crt_index_layer = 0;
			flat_input_depth = true;
			perceptron_indexes = new Array1DIfromD3(indexes, crt_index_layer, Field.getIndex(x, X_o, y, Y_o)); break;
		// nothing
		case Config.SOFTMAX: 
		}
		
		
		p[output_layer][Field.getIndex(x, X_o, y, Y_o)] =
			new Perceptron (filter[output_layer], D_i, xi, X_i, yi, Y_i, dilation, type, output_layer,
					perceptron_indexes, (output_layer == crt_index_layer), flat_input_depth );
		
	}
	
	@Override
	public void init(Array2DF input, int X_i, int Y_i, int D_i, Array2DF output, int X_o, int Y_o, int D_o, int x_pad,
			int y_pad, int stride, int dilation, Filter[] filter, Array2DF bias, Array3DF weights, int type,
			String name) {
		this.bias = bias;
		this.weights = weights; // new float[D_o][D_i][];
		
		//System.out.println("Do="+D_o+" Xo="+X_o+" Y_o="+Y_o+" FL="+filter[0].getFilterLinksNb());
		
		switch(type) {
		case Config.POOLMAX:
		case Config.POOLAVG:
			if (Config.FLAT_INDEXES_POOL) indexes = new Array3DI(1, X_o*Y_o, filter[0].getFilterLinksNb());
			else indexes = new Array3DI(D_o, X_o*Y_o, filter[0].getFilterLinksNb());
			break;
		case Config.CONVOLUTION:
		case Config.RELU:
		case Config.SIGMOID:
		case Config.TANH:
			if (Config.FLAT_INDEXES)
				indexes = new Array3DI(D_i, X_o*Y_o, filter[0].getWeightsNb());
			else
				indexes = new Array3DI(1, X_o*Y_o, filter[0].getWeightsNb());
			break;
		case Config.SOFTMAX: // nothing
		}
		
		p = new Perceptron[D_o][];
		int xi, yi;
		this.input = input;
		this.output = output;
		this.name = name;
		this.type = type;
		this.filter = filter;
		X_input = X_i; Y_input = Y_i; D_input = D_i;
		X_output = X_o; Y_output = Y_o; D_output = D_o;
		for (int output_layer = 0; output_layer < D_o; output_layer ++) {
			p[output_layer] = new Perceptron[X_o * Y_o];
		}
		
		yi = -y_pad + filter[0].y_half_low(); // index of first center y of field
		for (int y = 0; y < Y_o; y ++) {
			xi = -x_pad + filter[0].x_half_low();  // index of first center x of field
			for (int x = 0; x < X_o; x ++) {
				for (int output_layer = 0; output_layer < D_o; output_layer ++) {
					initPerceptron(output_layer, x, X_o, y, Y_o, D_i, xi, X_i, yi, Y_i, dilation);
				}
				xi += stride;
			}
			yi += stride;
		}
		/*
		IntStream.range(0,Y_o).parallel()
		.forEach(y->IntStream.range(0,X_o).parallel()
				.forEach(x -> IntStream.range(0,D_o).parallel()
						.forEach(output_layer ->
						initPerceptron(output_layer, x, X_o, y, Y_o, D_i, 
								-x_pad + filter[0].x_half_low() + stride*x, //xi, 
								X_i, 
								-y_pad + filter[0].y_half_low() + stride*y, //yi, 
								Y_i, dilation) ) ) );
		*/
		

		if (type == Config.SOFTMAX) softmax_aggregates = new float[output.getLength1()];
		
	}

	
	
	Filter filter[];  // hold information about geometry of activation field
	String name;      // a name for reference

	@Override
	public int getDepth_output() {
		return D_output;
	}

	@Override
	public int getDepth_input() {
		return D_input;
	}

	@Override
	public void setInput(Array2DF input) {
		this.input = input;
	}

	@Override
	public void update_GPU_weights() {
		throw new RuntimeException("This is a CPU implementation with no GPU use");
	}

	@Override
	public void convolute_with_CPU() {
		// Debug output possible
		if (Config.DEBUG_CPU_PERCEPTRONS) {
		   	for (int d = 0; d < input.getLength1(); d ++)
	    		for(int w = 0; w < input.getLength2(); w ++)
	    			System.out.println("Input: d="+d+", w="+ w+" => " + input.get(d,w));
		}
	   	
		//System.out.println("Convolute with CPU");
		if (type == Config.SOFTMAX) {convoluteSoftMaxCPU();}
		else {
			
			// A solution based on streams could be made parallel, but not really faster in tests
			//Arrays.stream(p).parallel().forEach(d -> Arrays.stream(d).forEach(k->k.convolute(input) ) );
			/*
			IntStream.range(0,p.length).parallel()
			.forEach(row->IntStream.range(0,p[row].length).parallel()
					.forEach(col -> output.set(row, col,
							p[row][col].convolute(input) ) ) );
			*/
			
			for (int d = 0; d < p.length; d ++)  {
				for (int k = 0; k < p[d].length; k ++) {
					//System.out.println("Convolute with CPU: d="+d+" k="+k);
					output.set(d, k,   p[d][k].convolute(input) );
				}
			}
			
		}
		
		
		
		// Debug output possible
		if (Config.DEBUG_CPU_PERCEPTRONS) {
		   	for (int d = 0; d < output.getLength1(); d ++)
	    		for(int w = 0; w < output.getLength2(); w ++)
	    			System.out.println("Result: d="+d+", w="+ w+" => " + output.get(d,w));
		}
	}
	
    /*
     * Solution avoiding repeat of aggregation
     */
    public void convoluteSoftMaxCPU() {
    	float aggregate = 0;
		for (int d = 0; d < p.length; d ++)  {
			for (int k = 0; k < p[d].length; k ++) {
				
				aggregate += output.set(d, k, (float) Math.exp(input.get(d, k)) );
			}
		}
		for (int d = 0; d < p.length; d ++)  {
			for (int k = 0; k < p[d].length; k ++) {
				output.set(d, k, output.get(d, k) / aggregate );
			}
		}
    }

	@Override
	public void convolute_with_GPU() {
		throw new RuntimeException("This is a CPU implementation with no GPU use");
	}

	@Override
	public void convolute_Default() {
		convolute_with_CPU();
	}

	@Override
	public boolean usesGPU() {
		return false;
	}

}