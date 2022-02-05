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
package cnn.util;

import cnn.Config;
import cnn.Field;

public
class Array3DI {
	private int dim1, dim2, dim3;
	private int[] data;
	private int[][][] data3;
	public Array3DI(int dim1, int dim2, int dim3) {
		this.dim1 = dim1;
		this.dim2 = dim2;
		this.dim3 = dim3;
		if (Config.FLAT_ARRAYS) data = new int[dim1 * dim2 * dim3];
		else data3 = new int[dim1][dim2][dim3];
		//System.out.println("Create RELU: length" + data.length +" dim1="+dim1);
	}
	public int get(int x1, int x2, int x3) {
		if (Config.FLAT_ARRAYS) return data[Field.getIndex(x1, dim1, x2, dim2, x3, dim3)];
		else return data3[x1][x2][x3];
	}
	public int set(int x1, int x2, int x3, int val) {
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
	public int[] getDataFlat() {
		return data;
	}
	public int[][][] getData3D() {
		return data3;
	}
	public int getLength1() {
		return dim1;
	}
	public int getLength2() {
		return dim2;
	}
	public int getLength3() {
		return dim3;
	}
}