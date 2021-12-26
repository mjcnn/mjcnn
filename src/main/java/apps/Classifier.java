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
package apps;

import cnn.*;
import cnn.Config.LayerConfig;

public class Classifier {
	static public void main (String args[]) {
		
		try {
			
			LayerConfig[] layers = Config.parseConfigVGG(null, args[0]);
			Array2DF input = new Array2DFimp ( layers[0].Do, layers[0].Xo*layers[0].Yo );
			Array2DF output = new Array2DFimp ( layers[layers.length-1].Do, layers[layers.length-1].Xo * layers[layers.length-1].Yo );

			CNN net = new CNN(layers, input, output);
			
			net.loadWeightsVGG(null, args[1]);
			
			net.setInput(input);
			net.classify_with_GPU();
			//net.classify_with_CPU();
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
