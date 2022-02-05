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
package aparapi_layers;

import com.aparapi.Kernel;
import com.aparapi.Range;

import cnn.Config;
import cnn.Field;
import cnn.Filter;
import cnn.LayerImplementation;
import cnn.Perceptron;
import cnn.util.Array1DI;
import cnn.util.Array1DIfromD3;
import cnn.util.Array2DF;
import cnn.util.Array3DF;
import cnn.util.Array3DI;

public
class LayerAparapi implements LayerImplementation {
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
	public void convolute_Default() {
		convolute_with_CPU();
	}
	
	Range range;       // used in launching GPU threads
	
	Kernel
	 kernel_linear, kernel_perceptron, //these two currently are handled identically, but 2nd might use 1st
	 kernel_pool, 
	 kernel_softmax, 
	 kernel_softmax_division, kernel_softmax_all; // these are not yet used, but are worth testing

	//public static GPU_Interface gpu = null;
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
	
	
	
	Filter filter[];  // hold information about geometry of activation field
	String name;      // a name for reference

	/**
	 * Old procedure that iterates sequentially over all perceptrons
	 */
	public void convolute_with_CPU () {
		//System.out.println("Convolute with CPU");
		if (type == Config.SOFTMAX) {convoluteSoftMaxCPU();}
		else {
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
		/*
	   	for (int d = 0; d < output.length; d ++)
    		for(int w = 0; w < output[0].length; w ++)
    			System.out.println("Result: " + output[d][w]);
	   	*/

	}
		
	public void update_GPU_weights() {
    	switch (type) {
    	case Config.CONVOLUTION: 
    		if (Config.FLAT_ARRAYS)
    			kernel_linear.put(weights.getDataFlat()).put(this.bias.getDataFlat()); 
    		else kernel_linear.put(weights.getData3D()).put(this.bias.getData2D()); 
    		break;
    	case Config.RELU:  
    	case Config.SIGMOID:  
    	case Config.TANH:  
    		if (Config.FLAT_ARRAYS)
    			kernel_perceptron.put(weights.getDataFlat()).put(this.bias.getDataFlat()); 
    		else kernel_perceptron.put(weights.getData3D()).put(this.bias.getData2D()); 
    		break;
    	case Config.POOLMAX:  
    	case Config.POOLAVG:  
    	case Config.SOFTMAX:  
    	}
    	return;
    }
	
	/**
	 * Initializing perceptron links in GPU Kernels
	 * Called from init_Kernels
	 */
	protected void init_indexes_GPU_Kernels() {
    	switch (type) {
     	case Config.CONVOLUTION:
    		if (Config.FLAT_ARRAYS) kernel_linear.put(this.indexes.getDataFlat());
    		else kernel_linear.put(this.indexes.getData3D());
    		break;
     	case Config.RELU:  
    	case Config.SIGMOID:  
    	case Config.TANH:
    		if (Config.FLAT_ARRAYS) kernel_perceptron.put(this.indexes.getDataFlat());
    		else kernel_perceptron.put(this.indexes.getData3D());
    		break;
    	case Config.POOLMAX:  
    	case Config.POOLAVG:
    		if (Config.FLAT_ARRAYS) kernel_pool.put(this.indexes.getDataFlat());
    		else kernel_pool.put(this.indexes.getData3D());
    		break;
    	case Config.SOFTMAX:  
    	}
    	return;
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
			indexes = new Array3DI(1, X_o*Y_o, filter[0].getWeightsNb()); break;
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
		
		range = Range.create2D(p.length, p[0].length);


		if (type == Config.SOFTMAX) softmax_aggregates = new float[output.getLength1()];
		
		init_Kernels();
		
	}

	/**
	 *  Constructor for many types of layers.
	 *  
	 * @param input:     [layer-depth][pixel-Field]
	 * @param X_i:       max_X input
	 * @param Y_i:       max_Y input
	 * @param D_i:       depth
	 * @param output:    [layer-depth][pixel-Field]
	 * @param X_o:
	 * @param Y_o:
	 * @param D_o:
	 * @param x_pad
	 * @param y_pad
	 * @param stride:    1 by default
	 * @param dilation:  0 by default, is number of elements between links on any dimension
	 * @param filter:    holds geometry of filter: margins needed
	 * @param bias:    [depth_output][1]   - one element per output filter
	 * @param weights: [depth_output][depth_input][field_X*field_Y]  - used in filter
	 * @param type:  Config.RELU/Config.SOFTMAX,...
	 * @param name: a decorative name, not used
	 */
	public LayerAparapi(
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
			flat_input_depth = false;
			perceptron_indexes = new Array1DIfromD3(indexes, crt_index_layer, Field.getIndex(x, X_o, y, Y_o)); break;
		// nothing
		case Config.SOFTMAX: 
		}
		
		
		p[output_layer][Field.getIndex(x, X_o, y, Y_o)] =
			new Perceptron (filter[output_layer], D_i, xi, X_i, yi, Y_i, dilation, type, output_layer,
					perceptron_indexes, (output_layer == crt_index_layer), flat_input_depth);
		
	}

	void init_Kernels() {
		if (Config.FLAT_ARRAYS)
		{
	    	switch (type) {
	    	case Config.CONVOLUTION: init_LinearKernelFlat(); break;
	    	case Config.RELU:  init_RELUKernelFlat(); break;
	    	case Config.SIGMOID:  init_SIGMKernelFlat(); break;
	    	case Config.TANH:  init_TANHKernelFlat(); break;
	    	case Config.POOLMAX:  init_POOLMAXKernelFlat(); break;
	    	case Config.POOLAVG:  init_POOLAVGKernelFlat(); break;
	    	case Config.SOFTMAX:  init_SoftMaxKernelFlat(); break; 
	    	}
		} else {
	    	switch (type) {
	    	case Config.CONVOLUTION: init_LinearKernel(); break;
	    	case Config.RELU:  init_RELUKernel(); break;
	    	case Config.SIGMOID:  init_SIGMKernel(); break;
	    	case Config.TANH:  init_TANHKernel(); break;
	    	case Config.POOLMAX:  init_POOLMAXKernel(); break;
	    	case Config.POOLAVG:  init_POOLAVGKernel(); break;
	    	//case Config.SOFTMAX:  init_SoftMaxKernel(); break; 
	    	}
		}
		init_indexes_GPU_Kernels();
    	return;
    }
	
	/**
	 * Currently only doing an exponential. Not working on MACOS because it uses doubles, unsupported
	 * 
	 * However, it could do all jobs with duplication of aggregation (as in init_SoftMaxKernel2)
	 */
	void init_SoftMaxKernelFlat() {
		final float _input[] = this.input.getDataFlat();
		final float _output[] = this.output.getDataFlat();
		final float _aggregate[] = this.softmax_aggregate;
		final float _aggregates[] = this.softmax_aggregates;
		final int INPUT_LENGTH = this.input.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
	
		if (false) { 
		kernel_softmax = new Kernel() {
			private float getV(float[] data, int x1, int dim1, int x2, int dim2) {
				return data[Field.getIndex(x1, dim1, x2, dim2)];
			}
			private void setV(float[] data, int x1, int dim1, int x2, int dim2, float val) {
				data[Field.getIndex(x1, dim1, x2, dim2)] = val;
			}
			
			@Override
		    public void run() {
				int od = getGlobalId(0);
				int ok = getGlobalId(1);
				int OD = getGlobalSize(0);
				int OK = getGlobalSize(1);
				setV ( _output, od, OD, ok, OK, 
						(float)Math.exp(getV ( _input, od, INPUT_LENGTH, ok, INPUT_0_LENGTH ) ) );
			}
		};
		kernel_softmax.setExplicit(true);
		
		
		/**
		 * The next kernel, if used, could do the division after an aggregation in CPU mode
		 */
		
		kernel_softmax_division = new Kernel() {
			private float getV(float[] data, int x1, int dim1, int x2, int dim2) {
				return data[Field.getIndex(x1, dim1, x2, dim2)];
			}
			private void setV(float[] data, int x1, int dim1, int x2, int dim2, float val) {
				data[Field.getIndex(x1, dim1, x2, dim2)] = val;
			}
			
			@Override
		    public void run() {
				int od = getGlobalId(0);
				int ok = getGlobalId(1);
				int OD = getGlobalSize(0);
				int OK = getGlobalSize(1);
				setV ( _output, od, OD, ok, OK, 
						getV (_output, od, OD, ok, OK)/_aggregate[0]);
			}
		};
		kernel_softmax_division.setExplicit(true);
		
		/**
		 * The next kernel, if used, would do everything and aggregation in logarithmic time
		 */

		kernel_softmax_all = new Kernel() {
			private float getV(float[] data, int x1, int dim1, int x2, int dim2) {
				return data[Field.getIndex(x1, dim1, x2, dim2)];
			}
			private void setV(float[] data, int x1, int dim1, int x2, int dim2, float val) {
				data[Field.getIndex(x1, dim1, x2, dim2)] = val;
			}
			
			@Override
		    public void run() {
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1); // will be 0
				int OK = getGlobalSize(1);
				boolean done = false;
				_aggregates[od] = (float)Math.exp(getV(_input, od, INPUT_LENGTH, ok, INPUT_0_LENGTH) );
				int bits = (int) (Math.log(OD)/Math.log(2) + 1);
				for (int k = 0; k < bits; k ++) {
					if (! done) {
						if ((od & (1 << k)) == 0) {
							if ((od | (1 << k)) >= OD) {
								_aggregate[0] += _aggregates[od];
								done = true;
							}
						} else {
							_aggregates[od & ~(1 << k)] += _aggregates[od];
							done = true;
						}
					}					
					localBarrier();
				}
				setV ( _output, od, OD, ok, OK,
						getV (_output, od, OD, ok, OK)/_aggregate[0]);
			}
		};
		kernel_softmax_all.setExplicit(true);
		}

		
	}
	
	/**
	 * Not used now. But possible implementation with duplication of aggregation. To me tested for efficiency!
	 * 
	 * The use of indexes can also be dispensed with since one certainly uses everything
	 */
	void init_SoftMaxKernelFlat2() {
		final float[] _input = this.input.getDataFlat();
		final float[] _output = this.output.getDataFlat();
		final int INPUT_LENGTH = this.input.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
	
		kernel_softmax = new Kernel() {
	   	    private int getIndex(int x, int X, int y, int Y) {
	   	    	return X*y + x;
	   	     }
			private float getV(float[] data, int x1, int dim1, int x2, int dim2) {
				return data[getIndex(x1, dim1, x2, dim2)];
			}
			private void setV(float[] data, int x1, int dim1, int x2, int dim2, float val) {
				data[getIndex(x1, dim1, x2, dim2)] = val;
			}

			@Override
		    public void run() {
				int od = getGlobalId(0);
				int ok = getGlobalId(1);
				int OD = getGlobalSize(0);
				int OK = getGlobalSize(1);
				
		    	float aggregate = 0; 	
				
		    	for (int id = 0; id < INPUT_LENGTH; id ++) {
		    		for (int ik = 0; ik < INPUT_0_LENGTH; ik ++) {    // filter_indexes.length
		    			aggregate += Math.exp( getV(_input, id, INPUT_LENGTH, ik, INPUT_0_LENGTH) );  // [id][filter_indexes[ik]]
		    		}
		    	}
		    	setV(_output, od, OD, ok, OK,
		    			(float)Math.exp( getV(_input, od, INPUT_LENGTH, ok, INPUT_0_LENGTH) )/aggregate );
			}
		};
		kernel_softmax.setExplicit(true);
		
	}
	void init_POOLMAXKernelFlat() {
		final int _indexes[] = this.indexes.getDataFlat();
		final float _input[] = this.input.getDataFlat();
		final float _output[] = this.output.getDataFlat();
		final int ID = this.input.getLength1();
		final int XD = this.indexes.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();
	
		kernel_pool = new Kernel() {
	   	    private int getIndex(int x, int X, int y, int Y) {
	   	    	return X*y + x;
	   	     }
			private float getV(float[] data, int x1, int dim1, int x2, int dim2) {
				return data[getIndex(x1, dim1, x2, dim2)];
			}
			private float getVI(float[] data, int x1, int dim1, int x2, int dim2) {
				if (Config.FLAT_INDEXES) return Config.FLAT_INDEXES_POOL?data[x1+x2]:data[x2];
				else return data[getIndex(x1, dim1, x2, dim2)];
			}
			private void setV(float[] data, int x1, int dim1, int x2, int dim2, float val) {
				data[getIndex(x1, dim1, x2, dim2)] = val;
			}

			private int getIndex(int x, int X, int y, int Y, int z, int Z) {
		    	int idx = X*(Y*z + y)+x;
		    	return idx;
		    }
			int getV(int[] data, int x1, int dim1, int x2, int dim2, int x3, int dim3) {
				return data[getIndex(x1, dim1, x2, dim2, x3, dim3)];
			}
			private int getAnchorScale(int x, int X, int y, int Y) {
				return X*Y;
			}
			private int getAnchorDisp(int x, int X, int y, int Y) {
				return X*y+x;
			}

			@Override
		    public void run() {
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
		    	//int filter_indexes[] = _indexes[od][ok]; //p[od][ok].idx;
				int anchor_scale=getAnchorScale(od, XD, ok, OK);//OD*OK;
				int anchor_displ=getAnchorDisp(od, XD, ok, OK);//OD*ok+0;
				int idx = anchor_displ;
				
		    	int ik = 0;
		    	float aggregate = getVI(_input, od, ID,
		    			//[getV(_indexes, od, OD, ok, OK, ik, FILTER_INDEXES_LENGTH)];
		    			_indexes[idx], INPUT_0_LENGTH);
    			idx += anchor_scale;
		    	for ( ik++ ; ik < FILTER_INDEXES_LENGTH; ik++) {
		    		float next = getVI(_input, od, ID,
			    			//[getV(_indexes, od, OD, ok, OK, ik, FILTER_INDEXES_LENGTH)];
		    				_indexes[idx], INPUT_0_LENGTH);
	    			idx += anchor_scale;
		    		if (next > aggregate)
		    			aggregate = next;
		    	}
		    	
		    	setV(_output, od, OD, ok, OK, aggregate);
			}
		};
		kernel_pool.setExplicit(true);
	}
	void init_POOLMAXKernel() {
		/*
    	float _input[][]=null;		//ID, INPUT_0_LENGHT
    	float _output[][]=null;		//OD, OK
    	int _indexes[][][]=null;	//OD, OK, FL
    	float _weights[][][]=null;	//OD, ID, FL
    	float _bias[][]=null;		//OD, 0
		 */
    	final int[][][] _indexes = this.indexes.getData3D();
		final float[][][] _weights = this.weights.getData3D();
		final float[][] _bias = this.bias.getData2D();
		final float _input[][] = this.input.getData2D();
		final float _output[][] = this.output.getData2D();
		final int ID = this.input.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();//_indexes[0][0].length;
		    
		kernel_pool = new Kernel() {
			@Override
			public void run () {	
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
		    	int filter_indexes[] = _indexes[od][ok]; //p[od][ok].idx;
				
		    	int ik = 0;
		    	float aggregate = _input[od][filter_indexes[ik]];
		    	for ( ik++ ; ik < FILTER_INDEXES_LENGTH; ik++) {
		    		float next = _input[od][filter_indexes[ik]];
		    		if (next > aggregate)
		    			aggregate = next;
		    	}
		    	_output[od][ok] = aggregate;
			}
		};
		kernel_pool.setExplicit(true);
	}
	
	void init_POOLAVGKernelFlat() {
		final int _indexes[] = this.indexes.getDataFlat();
		final float _input[] = this.input.getDataFlat();
		final float _output[] = this.output.getDataFlat();
		final int ID = this.input.getLength1();
		final int XD = this.indexes.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();//_indexes[0][0].length;
	
		kernel_pool = new Kernel() {
	   	    private int getIndex(int x, int X, int y, int Y) {
	   	    	return X*y + x;
	   	     }
			private float getV(float[] data, int x1, int dim1, int x2, int dim2) {
				return data[getIndex(x1, dim1, x2, dim2)];
			}
			private float getVI(float[] data, int x1, int dim1, int x2, int dim2) {
				if (Config.FLAT_INDEXES) return Config.FLAT_INDEXES_POOL?data[x1+x2]:data[x2];
				else return data[getIndex(x1, dim1, x2, dim2)];
			}
			private void setV(float[] data, int x1, int dim1, int x2, int dim2, float val) {
				data[getIndex(x1, dim1, x2, dim2)] = val;
			}

		    private int getIndex(int x, int X, int y, int Y, int z, int Z) {
		    	int idx = X*(Y*z + y)+x;
		    	return idx;
		    }
			int getV(int[] data, int x1, int dim1, int x2, int dim2, int x3, int dim3) {
				return data[getIndex(x1, dim1, x2, dim2, x3, dim3)];
			}
			private int getAnchorScale(int x, int X, int y, int Y) {
				return X*Y;
			}
			private int getAnchorDisp(int x, int X, int y, int Y) {
				return X*y+x;
			}

			@Override
		    public void run() {
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
		    	//int filter_indexes[] = _indexes[od][ok]; //p[od][ok].idx;
				int anchor_scale=getAnchorScale(od, XD, ok, OK);//OD*OK;
				int anchor_displ=getAnchorDisp(od, XD, ok, OK);//OD*ok+0;
				int idx = anchor_displ;
				
		    	int count = 0;
				
		    	int ik = 0;
		    	float aggregate = getVI(_input, od, ID,
		    			//[filter_indexes[ik]];
		    			//[getV(_indexes, od, OD, ok, OK, ik, FILTER_INDEXES_LENGTH)];
		    			_indexes[idx], INPUT_0_LENGTH);
    			idx += anchor_scale;
		    	for ( ik++ ; ik < FILTER_INDEXES_LENGTH; ik++) {
		    		aggregate += getVI(_input, od, ID,
		    			//[filter_indexes[ik]];
		    			//[getV(_indexes, od, OD, ok, OK, ik, FILTER_INDEXES_LENGTH)];
			    			_indexes[idx], INPUT_0_LENGTH);
	    			idx += anchor_scale;
		    		count ++;
		    	}
		    	setV(_output, od, OD, ok, OK, aggregate/count);
			}
		};
		kernel_pool.setExplicit(true);
	}
			
	void init_POOLAVGKernel() {
		/*
    	float _input[][]=null;		//ID, INPUT_0_LENGHT
    	float _output[][]=null;		//OD, OK
    	int _indexes[][][]=null;	//OD, OK, FL
    	float _weights[][][]=null;	//OD, ID, FL
    	float _bias[][]=null;		//OD, 0
		 */
    	final int[][][] _indexes = this.indexes.getData3D();
		final float[][][] _weights = this.weights.getData3D();
		final float[][] _bias = this.bias.getData2D();
		final float _input[][] = this.input.getData2D();
		final float _output[][] = this.output.getData2D();
		final int ID = this.input.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();//_indexes[0][0].length;
		    
		kernel_pool = new Kernel() {
			@Override
			public void run () {	
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
		    	int filter_indexes[] = _indexes[od][ok]; //p[od][ok].idx;
				
		    	int count = 0;
		    	int ik = 0;
		    	float aggregate = _input[od][filter_indexes[ik]];
		    	for ( ik++ ; ik < FILTER_INDEXES_LENGTH; ik++) {
		    		aggregate += _input[od][filter_indexes[ik]];
		    		count ++;
		    	}
		    	_output[od][ok] = aggregate/count;
			}
			
		};
		kernel_pool.setExplicit(true);
	}

	void init_RELUKernelFlat() {
		final int _indexes[] = this.indexes.getDataFlat();
		// System.out.println("Length indexes RELU:= "+_indexes.length);
		final float _weights[] = this.weights.getDataFlat();
		final float _bias[] = this.bias.getDataFlat();
		final float _input[] = this.input.getDataFlat();
		final float _output[] = this.output.getDataFlat();
		final int ID = this.input.getLength1();
		final int XD = this.indexes.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();//_indexes[0][0].length;
		
		//System.out.println("SZ _indexes: "+_indexes.length+" "+ _indexes[0].length+" "+_indexes[0][0].length);
		//System.out.println("SZ weights: "+_weights.length+" "+ _weights[0].length+" "+_weights[0][0].length);
		//System.out.println("SZ _bias: "+_bias.length+" "+ _bias[0].length);
		//System.out.println("SZ _input: "+_input.length+" "+ _input[0].length);
		//System.out.println("SZ _output: "+_output.length+" "+ _output[0].length);
	
		kernel_perceptron = new Kernel() {
	   	    private int getIndex(int x, int X, int y, int Y) {
	   	    	return X*y + x;
	   	     }
			private float getV(float[] data, int x1, int dim1, int x2, int dim2) {
				return data[getIndex(x1, dim1, x2, dim2)];
			}
			private float getVI(float[] data, int x1, int dim1, int x2, int dim2) {
				if (Config.FLAT_INDEXES) return data[x1+x2];
				else return data[getIndex(x1, dim1, x2, dim2)];
			}
			private void setV(float[] data, int x1, int dim1, int x2, int dim2, float val) {
				data[getIndex(x1, dim1, x2, dim2)] = val;
			}

			
		    private  int getIndex(int x, int X, int y, int Y, int z, int Z) {
		    	int idx = X*(Y*z + y)+x;
		    	return idx;
		    }
			private int getV(int[] data, int x1, int dim1, int x2, int dim2, int x3, int dim3) {
				return data[getIndex(x1, dim1, x2, dim2, x3, dim3)];
			}
			private int getAnchored(int[] data, int anchor_scale, int anchor_displ, int ik) {
				return data[anchor_scale * ik + anchor_displ];
			}
			private int getAnchorScale(int x, int X, int y, int Y) {
				return X*Y;
			}
			private int getAnchorDisp(int x, int X, int y, int Y) {
				return X*y+x;
			}
			@Override
		    public void run() {
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
				
		    	//int filter_indexes[] = _indexes[0][ok]; //p[od][ok].idx;
				int anchor_scale=getAnchorScale(0, XD, ok, OK);//XD*OK;
				int anchor_displ=getAnchorDisp(0, XD, ok, OK);//XD*ok+0;
				int idx = anchor_displ;
				
		    	float aggregate = _bias[od]; //[0]; // filter[od].getBias();
		    	
		    	for (int id = 0; id < ID; id ++) {     // filter[od].getDepth()
		    		//float __weights[] = _weights[od][id]; //filter[od].getWeightAsVector(id);
		    		// od+id*OD+ik*(OD*ID)
					int wanchor_scale=getAnchorScale(od, OD, id, ID);//OD*ID;
					int wanchor_displ=getAnchorDisp(od, OD, id, ID);//OD*id+od;
					int widx = wanchor_displ;
					
		    		for (int ik = 0; ik < FILTER_INDEXES_LENGTH; ik ++) {
		    			
		    			aggregate += 
		    					_weights[widx]//_weights[od][id] [ik] 
		    					* getVI(_input, id, ID,
		    							//_input[id][filter_indexes [ik]];
		    			    			//[getV(_indexes, 0, OD, ok, OK, ik, FILTER_INDEXES_LENGTH)];
		    							//[getAnchored(_indexes,anchor_scale, anchor_displ, ik)];
		    							_indexes[idx],
		    							INPUT_0_LENGTH);
		    			idx += anchor_scale;
		    			widx += wanchor_scale;
		    		}
		    		
		    	}
		    	
				setV(_output, od, OD, ok, OK, (float)aggregate>0?aggregate:0f);		
					
			}
		};
		kernel_perceptron.setExplicit(true);
	}
			
	
	void init_RELUKernel() {
		/*
    	float _input[][]=null;		//ID, INPUT_0_LENGHT
    	float _output[][]=null;		//OD, OK
    	int _indexes[][][]=null;	//OD, OK, FL
    	float _weights[][][]=null;	//OD, ID, FL
    	float _bias[][]=null;		//OD, 0
*/
    	final int[][][] _indexes = this.indexes.getData3D();
		// System.out.println("Length indexes RELU:= "+_indexes.length);
		final float[][][] _weights = this.weights.getData3D();
		final float[][] _bias = this.bias.getData2D();
		final float _input[][] = this.input.getData2D();
		final float _output[][] = this.output.getData2D();
		final int ID = this.input.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();//_indexes[0][0].length;
		    
		kernel_perceptron = new Kernel() {
			@Override
			public void run () {	
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
				
		    	int filter_indexes[] = _indexes[0][ok]; //p[od][ok].idx;
		    	float aggregate = _bias[od][0]; // filter[od].getBias();
		    	for (int id = 0; id < ID; id ++) {     // filter[od].getDepth()
		    		float __weights[] = _weights[od][id]; //filter[od].getWeightAsVector(id);
		    		for (int ik = 0; ik < FILTER_INDEXES_LENGTH; ik ++) {
		    			aggregate += 
		    					__weights[ik]//_weights[od][id] [ik] 
		    					* _input[id][filter_indexes [ik]];//[_indexes[0][ok][ik]];
		    		}		    		
		    	}
				_output[od][ok]= (float)aggregate>0?aggregate:0f;		
			}
			
		};
		kernel_perceptron.setExplicit(true);
	}
	void init_SIGMKernelFlat() {
		final int _indexes[] = this.indexes.getDataFlat();
		final float _weights[] = this.weights.getDataFlat();
		final float _bias[] = this.bias.getDataFlat();
		final float _input[] = this.input.getDataFlat();
		final float _output[] = this.output.getDataFlat();
		final int ID = this.input.getLength1();
		final int XD = this.indexes.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();//_indexes[0][0].length;
	
		kernel_perceptron = new Kernel() {
	   	    private int getIndex(int x, int X, int y, int Y) {
	   	    	return X*y + x;
	   	     }
			private float getV(float[] data, int x1, int dim1, int x2, int dim2) {
				return data[getIndex(x1, dim1, x2, dim2)];
			}
			private float getVI(float[] data, int x1, int dim1, int x2, int dim2) {
				if (Config.FLAT_INDEXES) return data[x1+x2];
				else return data[getIndex(x1, dim1, x2, dim2)];
			}
			private void setV(float[] data, int x1, int dim1, int x2, int dim2, float val) {
				data[getIndex(x1, dim1, x2, dim2)] = val;
			}

		    private int getIndex(int x, int X, int y, int Y, int z, int Z) {
		    	int idx = X*(Y*z + y)+x;
		    	return idx;
		    }
			int getV(int[] data, int x1, int dim1, int x2, int dim2, int x3, int dim3) {
				return data[getIndex(x1, dim1, x2, dim2, x3, dim3)];
			}

			private int getAnchorScale(int x, int X, int y, int Y) {
				return X*Y;
			}
			private int getAnchorDisp(int x, int X, int y, int Y) {
				return X*y+x;
			}
			@Override
		    public void run() {
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
		    	//int filter_indexes[] = _indexes[0][ok]; //p[od][ok].idx;
				int anchor_scale=getAnchorScale(0, XD, ok, OK);//OD*OK;
				int anchor_displ=getAnchorDisp(0, XD, ok, OK);//OD*ok+0;
				int idx = anchor_displ;
				
		    	float aggregate = _bias[od]; //[0]; // filter[od].getBias();
		    	for (int id = 0; id < ID; id ++) {     // filter[od].getDepth()
		    		//float __weights[] = _weights[od][id]; //filter[od].getWeightAsVector(id);
		    		
					int wanchor_scale=getAnchorScale(od, OD, id, ID);//OD*ID;
					int wanchor_displ=getAnchorDisp(od, OD, id, ID);//OD*id+od;
					int widx = wanchor_displ;
		    		
		    		for (int ik = 0; ik < FILTER_INDEXES_LENGTH; ik ++) {
		    			aggregate +=
		    					_weights[widx]//__weights[ik] 
		    					* getVI(_input, id, ID,
		    					//[filter_indexes[ik]];
    			    			//[getV(_indexes, 0, OD, ok, OK, ik, FILTER_INDEXES_LENGTH)];
		    					_indexes[idx], INPUT_0_LENGTH );
		    			idx += anchor_scale;
		    			widx += wanchor_scale;		    			
		    		}
		    	}
				setV(_output, od, OD, ok, OK, (float)(1 / (1 + Math.exp(-aggregate))));				
			}
		};
		kernel_pool.setExplicit(true);
	}
	void init_SIGMKernel() {
		/*
    	float _input[][]=null;		//ID, INPUT_0_LENGHT
    	float _output[][]=null;		//OD, OK
    	int _indexes[][][]=null;	//OD, OK, FL
    	float _weights[][][]=null;	//OD, ID, FL
    	float _bias[][]=null;		//OD, 0
		 */
    	final int[][][] _indexes = this.indexes.getData3D();
		final float[][][] _weights = this.weights.getData3D();
		final float[][] _bias = this.bias.getData2D();
		final float _input[][] = this.input.getData2D();
		final float _output[][] = this.output.getData2D();
		final int ID = this.input.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();//_indexes[0][0].length;
		   
		kernel_perceptron = new Kernel() {
			@Override
			public void run () {		    	
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
				
		    	int filter_indexes[] = _indexes[0][ok]; //p[od][ok].idx;
		    	float aggregate = _bias[od][0]; // filter[od].getBias();
		    	for (int id = 0; id < ID; id ++) {     // filter[od].getDepth()
		    		float __weights[] = _weights[od][id]; //filter[od].getWeightAsVector(id);
		    		for (int ik = 0; ik < FILTER_INDEXES_LENGTH; ik ++) {
		    			aggregate += 
		    					__weights[ik]//_weights[od][id] [ik] 
		    					* _input[id][filter_indexes [ik]];//[_indexes[0][ok][ik]];
		    		}		    		
		    	}
				_output[od][ok] = (float)(1 / (1 + Math.exp(-aggregate)));		
			}
		    
		};
		kernel_perceptron.setExplicit(true);
	}
	
	void init_TANHKernelFlat() {
		final int _indexes[] = this.indexes.getDataFlat();;
		final float _weights[] = this.weights.getDataFlat();
		final float _bias[] = this.bias.getDataFlat();
		final float _input[] = this.input.getDataFlat();
		final float _output[] = this.output.getDataFlat();
		final int ID = this.input.getLength1();
		final int XD = this.indexes.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();//_indexes[0][0].length;
	
		kernel_perceptron = new Kernel() {
	   	    private int getIndex(int x, int X, int y, int Y) {
	   	    	return X*y + x;
	   	     }
			private float getV(float[] data, int x1, int dim1, int x2, int dim2) {
				return data[getIndex(x1, dim1, x2, dim2)];
			}
			private float getVI(float[] data, int x1, int dim1, int x2, int dim2) {
				if (Config.FLAT_INDEXES) return data[x1+x2];
				else return data[getIndex(x1, dim1, x2, dim2)];
			}
			private void setV(float[] data, int x1, int dim1, int x2, int dim2, float val) {
				data[getIndex(x1, dim1, x2, dim2)] = val;
			}

		    private int getIndex(int x, int X, int y, int Y, int z, int Z) {
		    	int idx = X*(Y*z + y)+x;
		    	return idx;
		    }
			int getV(int[] data, int x1, int dim1, int x2, int dim2, int x3, int dim3) {
				return data[getIndex(x1, dim1, x2, dim2, x3, dim3)];
			}
			private int getAnchorScale(int x, int X, int y, int Y) {
				return X*Y;
			}
			private int getAnchorDisp(int x, int X, int y, int Y) {
				return X*y+x;
			}
			@Override
		    public void run() {
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
		    	//int filter_indexes[] = _indexes[0][ok]; //p[od][ok].idx;
				int anchor_scale=getAnchorScale(0, XD, ok, OK);//OD*OK;
				int anchor_displ=getAnchorDisp(0, XD, ok, OK);//OD*ok+0;
				int idx = anchor_displ;
				
				
		    	float aggregate = _bias[od]; //[0]; // filter[od].getBias();
		    	for (int id = 0; id < ID; id ++) {     // filter[od].getDepth()
		    		//float __weights[] = _weights[od][id]; //filter[od].getWeightAsVector(id);
		    		
					int wanchor_scale=getAnchorScale(od, OD, id, ID);//OD*ID;
					int wanchor_displ=getAnchorDisp(od, OD, id, ID);//OD*id+od;
					int widx = wanchor_displ;
		    		
		    		for (int ik = 0; ik < FILTER_INDEXES_LENGTH; ik ++) {
		    			aggregate +=
		    					_weights[widx] //__weights[ik]
		    					* getVI(_input, id, ID,
		    					//[filter_indexes[ik]];
    			    			//[getV(_indexes, 0, OD, ok, OK, ik, FILTER_INDEXES_LENGTH)];
				    			_indexes[idx], INPUT_0_LENGTH);
				    	idx += anchor_scale;
				    	widx += wanchor_scale;		    			
		    		}
		    	}
				setV(_output, od, OD, ok, OK, (float)Math.tanh(aggregate));				
			}
		};
		kernel_pool.setExplicit(true);
	}
	void init_TANHKernel() {
		/*
    	float _input[][]=null;		//ID, INPUT_0_LENGHT
    	float _output[][]=null;		//OD, OK
    	int _indexes[][][]=null;	//OD, OK, FL
    	float _weights[][][]=null;	//OD, ID, FL
    	float _bias[][]=null;		//OD, 0
		 */
    	final int[][][] _indexes = this.indexes.getData3D();
		final float[][][] _weights = this.weights.getData3D();
		final float[][] _bias = this.bias.getData2D();
		final float _input[][] = this.input.getData2D();
		final float _output[][] = this.output.getData2D();
		final int ID = this.input.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();//_indexes[0][0].length;
		    
		kernel_perceptron = new Kernel() {
			@Override
			public void run () {		    	
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
				
		    	int filter_indexes[] = _indexes[0][ok]; //p[od][ok].idx;
		    	float aggregate = _bias[od][0]; // filter[od].getBias();
		    	for (int id = 0; id < ID; id ++) {     // filter[od].getDepth()
		    		float __weights[] = _weights[od][id]; //filter[od].getWeightAsVector(id);
		    		for (int ik = 0; ik < FILTER_INDEXES_LENGTH; ik ++) {
		    			aggregate += 
		    					__weights[ik]//_weights[od][id] [ik] 
		    					* _input[id][filter_indexes [ik]];//[_indexes[0][ok][ik]];
		    		}		    		
		    	}
				_output[od][ok] = (float)Math.tanh(aggregate);		
			}
		};
		kernel_perceptron.setExplicit(true);
	}
	void init_LinearKernelFlat() {
		final int _indexes[] = this.indexes.getDataFlat();;
		final float _weights[] = this.weights.getDataFlat();
		final float _bias[] = this.bias.getDataFlat();
		final float _input[] = this.input.getDataFlat();
		final float _output[] = this.output.getDataFlat();
		final int ID = this.input.getLength1();
		final int XD = this.indexes.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();//_indexes[0][0].length;
		
	   	kernel_linear = new Kernel() {
	   	    private int getIndex(int x, int X, int y, int Y) {
	   	    	return X*y + x;
	   	     }

			private float getV(float[] data, int x1, int dim1, int x2, int dim2) {
				return data[getIndex(x1, dim1, x2, dim2)];
			}
			private float getVI(float[] data, int x1, int dim1, int x2, int dim2) {
				if (Config.FLAT_INDEXES) return data[x1+x2];
				else return data[getIndex(x1, dim1, x2, dim2)];
			}
			private void setV(float[] data, int x1, int dim1, int x2, int dim2, float val) {
				data[getIndex(x1, dim1, x2, dim2)] = val;
			}

		    private int getIndex(int x, int X, int y, int Y, int z, int Z) {
		    	int idx = X*(Y*z + y)+x;
		    	return idx;
		    }
			int getV(int[] data, int x1, int dim1, int x2, int dim2, int x3, int dim3) {
				return data[getIndex(x1, dim1, x2, dim2, x3, dim3)];
			}
			private int getAnchorScale(int x, int X, int y, int Y) {
					return X*Y;
			}
			private int getAnchorDisp(int x, int X, int y, int Y) {
					return X*y+x;
			}
			@Override
			public void run() {
				int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
		    	//int filter_indexes[] = _indexes[0][ok]; //p[od][ok].idx;
				int anchor_scale=getAnchorScale(0, XD, ok, OK);//OX*OK;
				int anchor_displ=getAnchorDisp(0, XD, ok, OK);//OX*ok+0;
				int idx = anchor_displ;
					
		    	float aggregate = _bias[od]; //[0]; // filter[od].getBias();
		    	for (int id = 0; id < ID; id ++) {     // filter[od].getDepth()
		    		//float __weights[] = _weights[od][id]; //filter[od].getWeightAsVector(id);
		    		
					int wanchor_scale=getAnchorScale(od, OD, id, ID);//OD*ID;
					int wanchor_displ=getAnchorDisp(od, OD, id, ID);//OD*id+od;
					int widx = wanchor_displ;
		    		
		    		for (int ik = 0; ik < FILTER_INDEXES_LENGTH; ik ++) {
		    			aggregate +=
		    					_weights[widx] //__weights[ik]
		    					* getVI(_input, id, ID,
	    					//[filter_indexes[ik]];
	 		    			//[getV(_indexes, 0, OD, ok, OK, ik, FILTER_INDEXES_LENGTH)];
	    					_indexes[idx], INPUT_0_LENGTH);
			   			idx += anchor_scale;
				    	widx += wanchor_scale;		    			
			    	}
			    }
				setV(_output, od, OD, ok, OK, aggregate);				
			}
		};
		kernel_linear.setExplicit(true);
	}
			
	void init_LinearKernel() {
		/*
    	float _input[][]=null;		//ID, INPUT_0_LENGHT
    	float _output[][]=null;		//OD, OK
    	int _indexes[][][]=null;	//OD, OK, FL
    	float _weights[][][]=null;	//OD, ID, FL
    	float _bias[][]=null;		//OD, 0
		 */
    	final int[][][] _indexes = this.indexes.getData3D();
		final float[][][] _weights = this.weights.getData3D();
		final float[][] _bias = this.bias.getData2D();
		final float _input[][] = this.input.getData2D();
		final float _output[][] = this.output.getData2D();
		final int ID = this.input.getLength1();
		final int INPUT_0_LENGTH = this.input.getLength2();
		final int FILTER_INDEXES_LENGTH = this.indexes.getLength3();//_indexes[0][0].length;
		    
		kernel_linear = new Kernel() {
			@Override
			public void run () {		    	
		    	int od = getGlobalId(0);
				int OD = getGlobalSize(0);
				int ok = getGlobalId(1);
				int OK = getGlobalSize(1);
		    	int filter_indexes[] = _indexes[0][ok]; //p[od][ok].idx;
					
		    	float aggregate = _bias[od][0]; // filter[od].getBias();
		    	for (int id = 0; id < ID; id ++) {     // filter[od].getDepth()
		    		float __weights[] = _weights[od][id]; //filter[od].getWeightAsVector(id);
		    				    		
		    		for (int ik = 0; ik < FILTER_INDEXES_LENGTH; ik ++) {
		    			aggregate +=
		    					__weights[ik]
		    					* _input[id][filter_indexes[ik]];
			    	}
			    }
				_output[od][ok] = aggregate;				
			}
			
		};
		kernel_linear.setExplicit(true);
	}
	
    public void convolute_with_GPU ( ) {
    	//System.out.println("Convolute GPU");
    	if (Config.FLAT_ARRAYS) {
	    	switch (type) {
	    	case Config.RELU:  
	    	case Config.SIGMOID: 
	    	case Config.TANH:  convolutePerceptronFlat(); break;
	    	case Config.CONVOLUTION: convoluteLinearFlat(); break;
	    	
	    	case Config.POOLMAX:  
	    	case Config.POOLAVG: convolutePoolFlat(); break;
	    		
	    	case Config.SOFTMAX: 
	    		// convoluteSoftMaxFlat(); break; // Math.exp not supported by aparapi
	    		 convoluteSoftMaxCPU(); break; // Math.exp not supported by aparapi
	    	}
    	} else {
	    	switch (type) {
	    	case Config.RELU:  
	    	case Config.SIGMOID: 
	    	case Config.TANH:  convolutePerceptron(); break;
	    	case Config.CONVOLUTION: convoluteLinear(); break;
	    	
	    	case Config.POOLMAX:  
	    	case Config.POOLAVG: convolutePool(); break;
	    		
	    	case Config.SOFTMAX: 
	    		 convoluteSoftMaxCPU(); break; // Math.exp not supported by aparapi
	    	}
    	}
    	/*
    	for (int d = 0; d < output.length; d ++)
    		for(int w = 0; w < output[0].length; w ++)
    			System.out.println("Result: " + output[d][w]);
    	*/
    	return;
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
    /**
     * convoluteSoftMax:
     * Not supported on MAcOS because Aparapi does not support Math.exp
     */
    public void convoluteSoftMaxFlat() {
		final float _input[] = this.input.getDataFlat();
		final float _output[] = this.output.getDataFlat();
		kernel_softmax.put(_input).execute(range).get(_output);
		
		float aggregate = 0;
		for (int i = 0 ; i < output.getLength1(); i ++) {
			aggregate += output.get(i, 0);
		}
		for (int i = 0 ; i < output.getLength1(); i ++) {
			output.set(i, 0, output.get(i, 0)/aggregate );
		}
    }
   
    public void convolutePoolFlat() {
		final float _input[] = this.input.getDataFlat();
		final float _output[] = this.output.getDataFlat();
		
		kernel_pool.put(_input).execute(range).get(_output);       	
		//System.out.println("Pool Execution mode = "+kernel_pool.getExecutionMode());
    }
 
    public void convolutePerceptronFlat() {
		final float _input[] = this.input.getDataFlat();
		final float _output[] = this.output.getDataFlat();
		
		kernel_perceptron.put(_input).execute(range).get(_output);   
		//kernel_perceptron.dispose();
		//System.out.println("Perc Execution mode = "+kernel_perceptron.getExecutionMode());
   }
    
	public void convoluteLinearFlat() {
		final float _input[] = this.input.getDataFlat();
		final float _output[] = this.output.getDataFlat();
		
		kernel_linear.put(_input).execute(range).get(_output);
		
		
		//System.out.println("Lin Execution mode = "+kernel_linear.getExecutionMode());
	}

    public void convolutePool() {
		final float _input[][] = this.input.getData2D();
		final float _output[][] = this.output.getData2D();
		
		kernel_pool.put(_input).execute(range).get(_output);       	
    }
 
    public void convolutePerceptron() {
		final float _input[][] = this.input.getData2D();
		final float _output[][] = this.output.getData2D();
		
		kernel_perceptron.put(_input).execute(range).get(_output);   
   }
    
	public void convoluteLinear() {
		final float _input[][] = this.input.getData2D();
		final float _output[][] = this.output.getData2D();
		
		kernel_linear.put(_input).execute(range).get(_output);
	}
	@Override
	public boolean usesGPU() {
		return true;
	}

}