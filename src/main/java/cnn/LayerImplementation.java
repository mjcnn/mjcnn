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

import cnn.util.Array2DF;
import cnn.util.Array3DF;

/**
 * 
 * @author Marius Silaghi
 * 
 * Interface for implementation of Layers with GPUs or CPUs
 */
public
interface LayerImplementation {
	boolean usesGPU();
	/**
	 * 
	 * @return perceptrons per output pixel
	 */
	int getDepth_output();
	
	/**
	 * @return
	 * perceptrons per input pixel
	 */
	
	int getDepth_input();

	/**
	 *  Set the array with data with dimensions raster_size*depth
	 * @param input
	 */
	void setInput(Array2DF input);

	
	/**  After depth is updated, it may need to be synchronized with GPU memory
	 * 
	 */
	void update_GPU_weights();

	/** Only one of the following may be implemented. 
	 * 
	 */
	void convolute_with_CPU();

	void convolute_with_GPU();

	/** Commonly this calls one of the above two implementations, considered faster
	 *
	 */
	void convolute_Default();
	
	public void init (
			Array2DF input, int X_i, int Y_i, int D_i,
			Array2DF output, int X_o, int Y_o, int D_o,
			int x_pad, int y_pad, int stride, int dilation,
			Filter filter[], Array2DF bias, Array3DF weights, int type, String name
			);
	
}