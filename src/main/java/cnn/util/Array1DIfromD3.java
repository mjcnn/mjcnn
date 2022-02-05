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
class Array1DIfromD3 implements  Array1DI{
	int x1, x2;
	Array3DI a3di;
	public Array1DIfromD3(Array3DI a3di, int x1, int x2) {
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
	public int getLength() {
		return a3di.getLength3();
	}
}