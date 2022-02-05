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
class Array1DFfrom3D implements Array1DF {
	Array3DF a3df;
	int x1, x2;
	public Array1DFfrom3D(Array3DF a3df, int x1, int x2) {
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
	public int getLength() {return a3df.getLength3();}	
}