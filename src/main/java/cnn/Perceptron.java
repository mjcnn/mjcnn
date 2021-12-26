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

public class Perceptron {
    Filter filter;
    Array1DI idx;
    int type;
    private int D_i;  // depths of inputs (to be used in initializing indexes)
    private int idx_depth; // Softmax output depth index of current perceptron in dense layer

    /**
     * 
     * @param _filter
     * @param D_i: depths input
     * @param x_center
     * @param X      Size of X for input
     * @param y_center
     * @param Y      Size of Y for input
     * @param dilation
     * @param type
     * @param O_depth:  Depth of the softmax output feed by this, or the pool depth input/output (0 valid on FLAT_INDEXES_POOL)
     * @param indexes 
     * @param init_indexes: Should we init the indexes here?
     */
    public Perceptron(Filter _filter, int D_i, int x_center, int X, int y_center, int Y, int dilation, int type, int O_depth, 
    		Array1DI indexes, boolean init_indexes) {
    	filter = _filter;
    	this.idx_depth = O_depth;
    	this.type = type;
    	int k = 0, _x, _y;
    	//idx = new int[filter.getX() * filter.getY()];
    	idx = indexes;
    	if (init_indexes) {
	    	for (
	    			int y = y_center - filter.y_half_low()*(1+dilation);
	    			y < y_center + filter.y_half_high()*(1+dilation);
	    			y += (1+dilation)
	    			)
	    	{
	    		if (y < 0) _y = 0; else _y = y;
	    		if (y >= Y) _y = Y - 1;
	    		for (
	    				int x = x_center - filter.x_half_low()*(1+dilation);
	    				x < x_center + filter.x_half_high()*(1+dilation);
	    				x += (1+dilation)
	    				)
	    		{
	    			if (x < 0) _x = 0; else _x = x;
	    			if (x >= X) _x = X - 1;
	    			
	    			if (Config.FLAT_INDEXES && Config.FLAT_ARRAYS) idx.set(k ++,  Field.getIndex(O_depth, D_i, _x, X, _y, Y));
	    			else idx.set(k ++,  Field.getIndex(_x, X, _y, Y));
	    		}
	    	}
    	}
    }
    
    /**
     * Perceptron level functions. Would not easily support GPU.
     */
    
    /**
     * Perform just a linear convolution of weights to input
     * at predefined indices as programmed in constructor
     * based on center and filter size/dilation
     * @param input
     * @return
     */
    public float convoluteSimple(Array2DF input) {
    	//System.out.println("Convolute Simple: ");
    	float aggregate = filter.getBias();
    	for (int d = 0; d < filter.getDepth(); d ++) {
    		Array1DF weights = filter.getWeightAsVector(d);
    		for (int k = 0; k < idx.length(); k++) {
    	    	//System.out.println("Convolute Simple : w="+weights[k]+" i="+input[d][idx[k]]);
    			aggregate += weights.get(k) * input.get(d, idx.get(k));
    		}
    	}
    	return aggregate;
    }
    
    public float convoluteRELU(Array2DF input) {
    	//System.out.println("Convolute RELU");
    	float aggregate = convoluteSimple(input);
    	if (aggregate < 0) return 0; /* relu the nonlinearity */
    	return aggregate;
    }
    
    public float convoluteSigmoid(Array2DF input) {
    	float aggregate = convoluteSimple(input);
    	return (float)(1 / (1 + Math.exp(-aggregate))); // sigmoid
    }
    
    public float convoluteTANH(Array2DF input) {
    	float aggregate = convoluteSimple(input);
    	return (float)Math.tanh(aggregate);
    }
    
    public float convolutePoolMax(Array2DF input) {
    	int d = filter.getOutputLayer();
    	assert (d == idx_depth);

    	int k = 0;
    	float aggregate = input.get(d, idx.get(k));
    	for ( k++ ; k < idx.length(); k++) {
    		aggregate = Math.max(input.get(d, idx.get(k)), aggregate);
    	}
    	return aggregate;
    }
    /**
     * POOL has filter associating inputs to variables, but has no weights.
     * @param input
     * @return
     */
    public float convolutePoolAvg(Array2DF input) {
    	float aggregate = 0; 
    	int count = 0;
    	int d = filter.getOutputLayer();
    	assert (d == idx_depth);
    	for (int k = 0; k < idx.length(); k++) {
    			aggregate += input.get(d, idx.get(k));
    			count ++;
    	}
    	return aggregate/count;
    }
    /**
     * Softmax has filter associating inputs to variables, but has no weights.
     * A previous simple conv layer can have weights.
     * 
     * TODO: Should be optimized factoring out the aggregation!
     * @param input
     * @return
     */
    public float convoluteSoftMax(Array2DF input) {
    	float aggregate = 0; 
    	for (int d = 0; d < filter.getDepth(); d ++) {
    		for (int k = 0; k < idx.length(); k++) {
    			aggregate += Math.exp( input.get(d, idx.get(k)) );
    		}
    	}
    	return (float)Math.exp(input.get(idx_depth, 0))/aggregate;
    }
    public float convolute(Array2DF input) {
    	//System.out.print("type="+Config.CNNtype(type));
    	switch (type) {
    	case Config.RELU: return convoluteRELU(input); 
    	case Config.SIGMOID: return convoluteSigmoid(input); 
    	case Config.TANH: return convoluteTANH(input); 
    	case Config.POOLMAX: return convolutePoolMax(input); 
    	case Config.POOLAVG: return convolutePoolAvg(input); 
    	case Config.CONVOLUTION: return convoluteSimple(input); 
    	case Config.SOFTMAX: return convoluteSoftMax(input); 
    	}
    	return 0;
    }
}
