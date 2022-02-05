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
class Array1DFfrom2D implements Array1DF {
	Array2DF a2df;
	int x1;
	public Array1DFfrom2D(Array2DF a2df, int x1) {
		this.a2df = a2df;
		this.x1 = x1;
	}
	public float get( int x2) {
		return a2df.get(x1, x2);
	}
	public float set( int x2, float val) {
		return a2df.set(x1, x2, val);
	}
	public int getLength() {return a2df.getLength2();}	
}