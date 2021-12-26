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

public
class Array2DFimp implements Array2DF {
	int dim1, dim2;
	
	// Switch the next two lines to move to 2D array implementation
	private float[] data;
	private float[][] data2;  
	
	public Array2DFimp(int dim1, int dim2) {
		this.dim1 = dim1;
		this.dim2 = dim2;
		if (Config.FLAT_ARRAYS) data = new float[dim1 * dim2];
		else data2 = new float[dim1][dim2];
	}
	public static float getV(float[] data, int x1, int dim1, int x2, int dim2) {
		if (Config.FLAT_ARRAYS) return data[Field.getIndex(x1, dim1, x2, dim2)];
		else throw new RuntimeException("flat"); //return data2[x1][x2];
	}
	public static float setV(float[] data, int x1, int dim1, int x2, int dim2, float val) {
		if (Config.FLAT_ARRAYS) return (data[Field.getIndex(x1, dim1, x2, dim2)] = val);
		else throw new RuntimeException("flat"); //return (data2[x1][x2] = val);
	}
	
	public float get(int x1, int x2) {
		if (Config.FLAT_ARRAYS) return getV(data,x1, dim1, x2, dim2);
		else return data2[x1][x2];
	}
	public float set(int x1, int x2, float val) {
		if (Config.FLAT_ARRAYS) return setV(data, x1, dim1, x2, dim2, val);
		else return (data2[x1][x2] = val);
	}
	
	
	public int length_1() {return dim1;}
	public int length_2() {return dim2;}
	@Override
	public Array1DF get1DF(int x1) {
		//throw new RuntimeException("Not implemented Array2DF");
		return new Array1DFfrom2D(this, x1);
	}
	@Override
	public float[] getDataFlat() {
		return data;
	}
	@Override
	public float[][] getData2D() {
		return data2;
	}	
}