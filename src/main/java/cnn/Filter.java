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

import cnn.util.Array1DF;
import cnn.util.Array2DF;
import cnn.util.Array2DFimp;
import cnn.util.Array3DF;

public class Filter {
    public int x_size, y_size;
    private int depth, output_layer;
    private Array1DF bias;       // size 1
    private Array2DF weight;   // [input_depth][index_raster_activation_field]

    public int x_half_low() { return (x_size-1)/2; }
    public int x_half_high() { return (x_size)/2; }
    public int y_half_low() { return (y_size-1)/2; }
    public int y_half_high() { return (y_size)/2; }
    public int getX() {return x_size;}
    public int getY() {return y_size;}
    public int getDepth() {return depth;}
    public int getOutputLayer() {return output_layer;}

    /**
     * Constructor for softmax/pool-like layers, having no weights
     * @param X
     * @param Y
     * @param depth
     */
    public Filter(int X, int Y, int depth) {
    	x_size = X;
    	y_size = Y;
    	this.depth = depth; // input depth
    }
    /**
     * Constructor for filters of convolution layers
     * @param X
     * @param Y
     * @param depth of input
     * @param level at which it works
     */
    public Filter(int X, int Y, int depth, int output_layer, Array1DF bias, Array2DF weights) {
    	x_size = X;
    	y_size = Y;
    	this.depth = depth;
    	this.output_layer = output_layer;
    	this.bias = bias;
    	this.weight = weights;
    }

    public float getBias() {
    	return bias.get(0);
    }
    public void setBias(float value) {
    	bias.set(0, value);
    }
    /**
     * The weight for input (x,y) of the feature 'd' perceptron
     * @param x
     * @param y
     * @param d
     * @return
     */
    public float getWeight(int x, int y, int d) {
    	return weight.get(d, Field.getIndex(x, x_size, y, y_size) );
    }
    public void setWeight(int x, int y, int d, float val) {
    	weight.set(d, Field.getIndex(x, x_size, y, y_size), val);
    }
    /**
     * Setter-getters using raster mapping to one dimensional vector
     * @param d
     * @param w
     * @param value
     */
	public void setWeight(int d, int w, Float value) {
		weight.set(d, w, value);
	}
	public float getWeight(int d, int w) {
		return weight.get(d, w);
	}

    /**
     * Returns the underlying weights vector
     * @param d
     * @return
     */
    public Array1DF getWeightAsVector(int d) {
    	if (weight == null) return null;
    	return weight.get1DF(d);
    }
    public Array2DF getWeightAsVectors() {
    	return weight;
    }
    /**
     * Create an array of filters to be used for all the features of an output pixel
     * @param outputs : tells the number of dimensions of the output pixel
     * 	(returned array dimension)
     * @param X
     * @param Y
     * @param depth: tells the number of dimensions of the input pixel
     * 
     * @return
     */
    public static Filter[] getArrayPoolFilter(int outputs, int X, int Y, int depth) {
    	Filter[] filters = new Filter[outputs];
    	for (int k = 0; k < outputs; k ++) {
    		filters[k] = new Filter(X, Y, depth);
    	}
    	return filters;
    }
    
    public static Filter[] getArrayFilter(int outputs, int X, int Y, int depth, int level, Array2DF bias[], Array3DF weights[]) {
    	
    	Filter[] filters = new Filter[outputs];
    	bias[level] = new Array2DFimp(outputs, 1);//float[outputs][1];
    	weights[level] = new Array3DF(outputs, depth, X*Y);
    		
    	for (int output_layer = 0; output_layer < outputs; output_layer ++) {
    		filters[output_layer] =
    			new Filter(X, Y, depth, output_layer, bias[level].get1DF(output_layer), weights[level].getArray2DF(output_layer));
    	}
    	return filters;
    }
    
    public int getFilterLinksNb() {
    	return x_size * y_size;
    }
    
    public int getWeightsNb() {
    	if (weight == null) return 0;
		return weight.getLength2();
	}
    
}
