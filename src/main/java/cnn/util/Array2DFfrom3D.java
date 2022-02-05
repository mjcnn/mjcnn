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

public
class Array2DFfrom3D implements Array2DF {
	Array3DF a3df;
	int x1;
	public Array2DFfrom3D(Array3DF a3df, int x1) {
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
	public int getLength1() {return a3df.getLength2();}
	public int getLength2() {return a3df.getLength3();}
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
	@Override
	public float get(int i) {
		throw new RuntimeException("Underlying Data not 1D");
	}	
}