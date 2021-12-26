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

/**
 * 
 * @author Marius Silaghi
 * Used for standardizing raster access in two dimensional rasters represented as one vector
 */
public class Field {
	/**
	 * 
	 * @param x
	 * @param X: max dimension on x
	 * @param y
	 * @param Y: max dimension on y
	 * @return
	 */
    public static int getIndex(int x, int X, int y, int Y) {
	assert(0<=x && x<X && 0<=y && y<Y);
	int idx = X*y + x;
	// System.out.println(""+x+" "+y+" -> "+idx);
	return idx;
    }

    public static int getIndex(int x, int X, int y, int Y, int z, int Z) {
    	assert(0<=x && x<X && 0<=y && y<Y && 0<=z && z<Z);
    	int idx = X*(Y*z + y)+x;
    	// System.out.println(""+x+" "+y+" -> "+idx);
    	return idx;
    }
    
}
class Array3DF {
	int dim1, dim2, dim3;
	private float[] data;
	private float[][][] data3;
	Array3DF(int dim1, int dim2, int dim3) {
		this.dim1 = dim1;
		this.dim2 = dim2;
		this.dim3 = dim3;
		if (Config.FLAT_ARRAYS) data = new float[dim1 * dim2 * dim3];
		else data3 = new float[dim1][dim2][dim3];
	}
	float getV(float[] data, int x1, int dim1, int x2, int dim2, int x3, int dim3) {
		if (Config.FLAT_ARRAYS) return data[Field.getIndex(x1, dim1, x2, dim2, x3, dim3)];
		else throw new RuntimeException("flat"); 
	}
	float setV(float[] data, int x1, int dim1, int x2, int dim2, int x3, int dim3, float val) {
		if (Config.FLAT_ARRAYS) return (data[Field.getIndex(x1, dim1, x2, dim2, x3, dim3)] = val);
		else throw new RuntimeException("flat"); 
	}
	float get(int x1, int x2, int x3) {
		if (Config.FLAT_ARRAYS) return getV(getDataFlat(), x1, dim1, x2, dim2, x3, dim3);
		else return (data3[x1][x2][x3]);
	}
	float set(int x1, int x2, int x3, float val) {
		if (Config.FLAT_ARRAYS) return setV(getDataFlat(), x1, dim1, x2, dim2, x3, dim3, val);
		else return (data3[x1][x2][x3] = val);
	}
	public Array2DF getArray2DF(int output_layer) {
		return new Array2DFfrom3D(this, output_layer);
	}
	float[] getDataFlat() {
		return data;
	}
	float[][][] getData3D() {
		return data3;
	}
}
class Array1DFfrom3D implements Array1DF {
	Array3DF a3df;
	int x1, x2;
	Array1DFfrom3D(Array3DF a3df, int x1, int x2) {
		this.a3df = a3df;
		this.x1 = x1;
		this.x2 = x2;
	}
	public float get(int x3) {
		return a3df.get(x1, x2, x3);
				//getDataFlat()[Field.getIndex(x1, a3df.dim1, x2, a3df.dim2, x3, a3df.dim3)];
	}
	public float set(int x3, float val) {
		return a3df.set(x1, x2, x3, val);
				//(a3df.getDataFlat()[Field.getIndex(x1, a3df.dim1, x2, a3df.dim2, x3, a3df.dim3)] = val);
	}
	public int length() {return a3df.dim3;}	
}

class Array1DFfrom2D implements Array1DF {
	Array2DF a2df;
	int x1;
	Array1DFfrom2D(Array2DF a2df, int x1) {
		this.a2df = a2df;
		this.x1 = x1;
	}
	public float get( int x2) {
		return a2df.get(x1, x2);
	}
	public float set( int x2, float val) {
		return a2df.set(x1, x2, val);
	}
	public int length() {return a2df.length_2();}	
}

class Array2DFfrom3D implements Array2DF {
	Array3DF a3df;
	int x1;
	Array2DFfrom3D(Array3DF a3df, int x1) {
		this.a3df = a3df;
		this.x1 = x1;
	}
	public float get(int x2, int x3) {
		return a3df.get(x1, x2, x3);
				//.getDataFlat()[Field.getIndex(x1, a3df.dim1, x2, a3df.dim2, x3, a3df.dim3)];
	}
	public float set(int x2, int x3, float val) {
		return a3df.set(x1, x2, x3, val);
				//(a3df.getDataFlat()[Field.getIndex(x1, a3df.dim1, x2, a3df.dim2, x3, a3df.dim3)] = val);
	}
	public int length_1() {return a3df.dim2;}
	public int length_2() {return a3df.dim3;}
	@Override
	public Array1DF get1DF(int x2) {
		return new Array1DFfrom3D(a3df, x1, x2);
	}
	@Override
	public float[] getDataFlat() {
		throw new RuntimeException("Underlying data not 2D");
		//return null;
	}
	@Override
	public float[][] getData2D() {
		throw new RuntimeException("Underlying data not 2D");
		//return null;
	}	
}

class Array3DI {
	private int dim1, dim2, dim3;
	private int[] data;
	private int[][][] data3;
	Array3DI(int dim1, int dim2, int dim3) {
		this.dim1 = dim1;
		this.dim2 = dim2;
		this.dim3 = dim3;
		if (Config.FLAT_ARRAYS) data = new int[dim1 * dim2 * dim3];
		else data3 = new int[dim1][dim2][dim3];
		//System.out.println("Create RELU: length" + data.length +" dim1="+dim1);
	}
	int get(int x1, int x2, int x3) {
		if (Config.FLAT_ARRAYS) return data[Field.getIndex(x1, dim1, x2, dim2, x3, dim3)];
		else return data3[x1][x2][x3];
	}
	int set(int x1, int x2, int x3, int val) {
		if (Config.FLAT_ARRAYS) return (data[Field.getIndex(x1, dim1, x2, dim2, x3, dim3)] = val);
		else return (data3[x1][x2][x3] = val);
	}
	static int getV(int[] data, int x1, int dim1, int x2, int dim2, int x3, int dim3) {
		if (Config.FLAT_ARRAYS) return data[Field.getIndex(x1, dim1, x2, dim2, x3, dim3)];
		else throw new RuntimeException("flat");
	}
	static void setV(int[] data, int x1, int dim1, int x2, int dim2, int x3, int dim3, int val) {
		if (Config.FLAT_ARRAYS) data[Field.getIndex(x1, dim1, x2, dim2, x3, dim3)] = val;
		else throw new RuntimeException("flat");
	}
	public int length3() {
		return dim3;
	}
	public int[] getDataFlat() {
		return data;
	}
	public int[][][] getData3D() {
		return data3;
	}
	public int length_1() {
		return dim1;
	}
}
interface Array1DI {
	int get(int x3);
	void set(int x3, int val);
	int length();
}
class Array1DIfromD3 implements  Array1DI{
	int x1, x2;
	Array3DI a3di;
	Array1DIfromD3(Array3DI a3di, int x1, int x2) {
		this.a3di = a3di;
		this.x1 = x1;
		this.x2 = x2;
	}
	public int get(int x3) {
		return a3di.get(x1, x2, x3);
	}
	public void set(int x3, int val) {
		a3di.set(x1, x2, x3, val);
	}
	@Override
	public int length() {
		return a3di.length3();
	}
}

