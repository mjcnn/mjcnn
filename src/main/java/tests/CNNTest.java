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
package tests;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import cnn.CNN;
import cnn.Config;
import cnn.Config.LayerConfig;
import cnn.util.Array2DF;
import cnn.util.Array2DFimp;

class CNNTest {
	private static final int INDEX_NOT_FOUND = -1;
	private static final String EMPTY = "";
	String w1="{"
			+ "'weights':["
			+ "  'level1':["
			+ "    'filter1':"
			+ "     [0,2,5,8,5],"
			+ "    'filter2':"
			+ "     [0,2,5,8,5]],"
			+ "  'level2':["
			+ "    'filter1':"
			+ "     [0,2,5,8,5],"
			+ "    'filter2':"
			+ "     [0,2,5,8,5]],"
			+ "   ]"
			+ "	 ]"
			+ "}";	
	String w2 = "{'weights':[\n"
			+ "'layer0':[\n"
			+ "	'filter0_0':[0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		], \n"
			+ "	'filter0_1':[0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 2.3, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		], \n"
			+ "	'filter0_2':[8.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0\n"
			+ "		]]]}";
	String w_xor = ""
			+ "{'weights':[\n"
			+ "'level1':[\n"
			+ " 'filterAND':[-1, 1, 1],\n"
			+ " 'filterOR': [0, 1, 1]\n"
			+ " ],\n"
			+ "'outputs':[\n"
			+ " 'filterXOR':[0, -2, 1]"
			+ " ]\n"
			+ "]}";
	String structure_one_level =
		    " {'levels':["
		    	    + "'level1':"
		    	    + "  { "
		    	    + "    'type':'INPUT',"
		    	    + "    'xo':256,"
		    	    + "    'yo':256,"
		    	    + "    'do':3"
		    	    + "  },"
		    	    + "'level2':"
		    	    + "  { "
		    	    + "    'type':'RELU',"
		    	    + "    'xo':256,"
		    	    + "    'yo':256,"
		    	    + "    'do':3,"
		    	    + "    'fx':5,"
		    	    + "    'fy':5,"
		    	    + "    'px':2,"
		    	    + "    'py':2,"
		    	    + "    'stride':1,"
		    	    + "    'dilation':0"
		    	    + "  }"
		    	    + " ]}";

	String structure_xor =
					"{'levels':["
					+ " 'input':"
					+ "  {"
					+ "  'type':'INPUT',"
					+ "   'xo':1,"
					+ "   'yo':1,"
					+ "   'do':2"
					+ "  },"
					+ " 'level1':"
					+ "  {"
					+ "   'type':'RELU',"
					+ "   'xo':1,"
					+ "   'yo':1,"
					+ "   'do':2,"
		    	    + "   'fx':1,"
		    	    + "   'fy':1"
					+ "  },"
					+ " 'outputs':"
					+ "  {"
					+ "   'type':'RELU',"
					+ "   'xo':1,"
					+ "   'yo':1,"
					+ "   'do':1,"
		    	    + "   'fx':1,"
		    	    + "   'fy':1"
					+ "  }"
					+ " ]}";

	@Test
	void testXOR() {
		try {
			LayerConfig[] layers =  Config.parseConfig(structure_xor, null);
			Array2DF input = new Array2DFimp(layers[0].Do, layers[0].Xo*layers[0].Yo );
			Array2DF output = new Array2DFimp ( layers[layers.length-1].Do, layers[layers.length-1].Xo * layers[layers.length-1].Yo );
			CNN net = new CNN(layers, input, output);
					
			String fname = null;
			net.loadWeights(w_xor, fname);
			
	input.set (0, 0, 0); 
	input.set (1, 0, 1); 
	net.setInput(input);
	// Array2DF result = net.classify_with_GPU();
	Array2DF result = net.classify_with_CPU();
	System.out.println("XOR:"+input.get(0, 0) +" ^ "+input. get(1, 0) );
	for (int d = 0; d < result.getLength1(); d ++) {
		for (int r = 0; r < result.getLength2(); r ++) {
				System.out.print(result.get(d, r) + " ");
		}
		System.out.println(" ");
	}
	assertEquals(true, result.get(0, 0) > 0);
	
	input.set (0, 0, 1); 
	input.set (1, 0, 0); 
	net.setInput(input);
	//result = net.classify_with_GPU();
	//float[][] 
	result = net.classify_with_CPU();
	System.out.println("XOR:"+input.get(0, 0) +" ^ "+input. get(1, 0) );
	for (int d = 0; d < result.getLength1(); d ++) {
		for (int r = 0; r < result.getLength2(); r ++) {
				System.out.print(result.get(d, r) + " ");
		}
		System.out.println(" ");
	}
	assertEquals(true, result.get(0, 0) >0);
	
	input.set (0, 0, 0); 
	input.set (1, 0, 0); 
	net.setInput(input);
	//result = net.classify_with_GPU();
	result = net.classify_with_CPU();
	System.out.println("XOR:"+input.get(0, 0) +" ^ "+input. get(1, 0) );
	for (int d = 0; d < result.getLength1(); d ++) {
		for (int r = 0; r < result.getLength2(); r ++) {
				System.out.print(result.get(d, r) + " ");
		}
		System.out.println(" ");
	}
	assertEquals(true, result.get(0, 0) ==0);
	
	
	input.set (0, 0, 1); 
	input.set (1, 0, 1); 
	net.setInput(input);
	//result = net.classify_with_GPU();
	result = net.classify_with_CPU();
	System.out.println("XOR:"+input.get(0, 0) +" ^ "+input. get(1, 0) );
	for (int d = 0; d < result.getLength1(); d ++) {
		for (int r = 0; r < result.getLength2(); r ++) {
			System.out.print(result.get(d, r) + " ");
		}
		System.out.println(" ");
	}
			assertEquals(false, result.get(0, 0) > 0);
		} catch(Exception e) {e.printStackTrace();}		
	}

	@Test
	void test() {
		try {
			LayerConfig[] layers =  Config.parseConfig(structure_one_level, null);
			Array2DF input = new Array2DFimp ( layers[0].Do, layers[0].Xo*layers[0].Yo );
			Array2DF output = new Array2DFimp ( layers[layers.length-1].Do, layers[layers.length-1].Xo * layers[layers.length-1].Yo );
			CNN net = new CNN(layers, input, output);
			
			String fname = "weights.w";
			net.loadWeights(w2, fname);
			String s = net.storeWeightsNative(fname);
			//System.out.println(s);
			//System.out.println("slen="+s.length());
			//System.out.println("sylen="+w2.length());
			System.out.println(difference(w2,s));
			assertEquals(w2.equals(s), true);
			
			net.setInput(input);
			//net.classify_with_GPU();
			net.classify_with_CPU();
		} catch(Exception e) {e.printStackTrace();}		
	}

	public static String difference(String str1, String str2) {
	    if (str1 == null) {
	        return str2;
	    }
	    if (str2 == null) {
	        return str1;
	    }
	    int at = indexOfDifference(str1, str2);
	    if (at == INDEX_NOT_FOUND) {
	        return EMPTY;
	    }
	    return str2.substring(at);
	}

	public static int indexOfDifference(CharSequence cs1, CharSequence cs2) {
	    if (cs1 == cs2) {
	        return INDEX_NOT_FOUND;
	    }
	    if (cs1 == null || cs2 == null) {
	        return 0;
	    }
	    int i;
	    for (i = 0; i < cs1.length() && i < cs2.length(); ++i) {
	        if (cs1.charAt(i) != cs2.charAt(i)) {
	            break;
	        }
	    }
	    if (i < cs2.length() || i < cs1.length()) {
	        return i;
	    }
	    return INDEX_NOT_FOUND;
	}
}
