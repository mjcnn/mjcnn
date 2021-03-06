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
    	if (!(0<=x && x<X && 0<=y && y<Y && 0<=z && z<Z)) {
    		System.out.println("x="+x+"/"+X+" y="+y+"/"+Y+" z="+z+"/"+Z);
    	}
    	assert(0<=x && x<X && 0<=y && y<Y && 0<=z && z<Z);
    	int idx = X*(Y*z + y)+x;
    	// System.out.println(""+x+" "+y+" -> "+idx);
    	return idx;
    }
    
}

