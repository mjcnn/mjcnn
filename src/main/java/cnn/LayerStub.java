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

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import cnn.util.Array2DF;
import cnn.util.Array3DF;

/**
 * 
 * @author Marius Silaghi
 *
 */
public class LayerStub implements LayerImplementation{
	public static  Class<? extends LayerImplementation> layer_class = LayerCPU.class;
	LayerImplementation layer_implementation = null;

	public LayerStub(LayerImplementation li) {layer_implementation = li;}

	/**
	 * This constructor can be used only if the current layer_class has a unique constructor 
	 * and that constructor has the same parameters as this constructor.
	 * 
	 * @param input
	 * @param X_i
	 * @param Y_i
	 * @param D_i
	 * @param output
	 * @param X_o
	 * @param Y_o
	 * @param D_o
	 * @param x_pad
	 * @param y_pad
	 * @param stride
	 * @param dilation
	 * @param filter
	 * @param bias
	 * @param weights
	 * @param type
	 * @param name
	 */
	public LayerStub(
			Array2DF input, int X_i, int Y_i, int D_i,
			Array2DF output, int X_o, int Y_o, int D_o,
			int x_pad, int y_pad, int stride, int dilation,
			Filter filter[], Array2DF bias, Array3DF weights, int type, String name
			) {
		
		try {
			Constructor<?>[] constructors = layer_class.getDeclaredConstructors();
			if (constructors.length != 1) throw new RuntimeException("Using implementation with too many constructors");
			layer_implementation = (LayerImplementation) constructors[0].newInstance(
					input,  X_i,  Y_i,  D_i,
					 output,  X_o,  Y_o,  D_o,
					 x_pad,  y_pad,  stride,  dilation,
					 filter,  bias,  weights,  type, name);
		} catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException
				| SecurityException e) {
			e.printStackTrace();
		}
		
	}
	
	public static void setLayerClass(Class<? extends LayerImplementation> lc) {
		layer_class = lc;
	}
	
	public void setLayerImplementation(LayerImplementation li) {
		layer_implementation = li;
	}

	public int getDepth_output() {
		return layer_implementation.getDepth_output();
	}

	public int getDepth_input() {
		return layer_implementation.getDepth_input();
	}

	public void setInput(Array2DF input) {
		layer_implementation.setInput(input);
	}

	public void update_GPU_weights() {
		layer_implementation.update_GPU_weights();
	}

	public void convolute_with_CPU() {
		layer_implementation.convolute_with_CPU();
	}

	public void convolute_with_GPU() {
		layer_implementation.convolute_with_GPU();
	}
	public void convolute_Default() {
		layer_implementation.convolute_Default();
	}

	@Override
	public void init(Array2DF input, int X_i, int Y_i, int D_i, Array2DF output, int X_o, int Y_o, int D_o, int x_pad,
			int y_pad, int stride, int dilation, Filter[] filter, Array2DF bias, Array3DF weights, int type,
			String name) {
		
		layer_implementation.init (
				input,  X_i,  Y_i,  D_i,
				 output,  X_o,  Y_o,  D_o,
				 x_pad,  y_pad,  stride,  dilation,
				 filter,  bias,  weights,  type, name);
		
	}

	@Override
	public boolean usesGPU() {
		return layer_implementation.usesGPU();
	}
}
