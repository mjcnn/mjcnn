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
package json;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Hashtable;

import cnn.CNN;
import cnn.Config;

@Deprecated
public class JSON {
	Object levels[];
	
	
	public static void loadWeights(CNN c) {
		String sweights =
"{"
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
		parseJSONWeights(c, sweights);
		//Item i = (Item) parseDeepJSON(sweights);
		//dumpDeepJSON(i);
	}
	
	static void dumpWeights(Hashtable<String,ArrayList<ArrayList<String>>> s) {
		for (String l : s.keySet()) {
			ArrayList<ArrayList<String>> lw = s.get(l);
			System.out.println("Dump\n" + l + ":");
			for (ArrayList<String> f : lw) {
				for (String w : f) {
					System.out.println(" \t"+w);
				}
				System.out.println();
			}
		}
		
	}

	static void parseJSONWeights(CNN net, String fname) {
		int curIndex=0;
		int nestedParentheses=0;
		int nestedSQParentheses=0;
		int subbranch = -1, branch = -1, property = -1, branches = 0;
		final int LEVELS = 0;
		final int LEVEL = 1;
		final int WEIGHTS = 2;

		final int XO = 0;
		final int YO = 1;
		final int DO = 2;
		final int FX = 3;
		final int FY = 4;
		final int PX = 5;
		final int PY = 6;
		final int STRIDE = 7;
		final int DILATION = 8;
		final int TYPE = 9;
		String Xo = null, Yo = null, Do = null, 
				Fx = null, Fy = null, 
				Px = null, Py = null,
				stride = null, dilation = null, type = null;

		int keyOrValue=0;//nu stiu daca incepe cu key sau value; 0 key, 1 value
		ArrayList<String> levelnames = null;
		ArrayList<Object> levelsarray = null;

		ArrayList<String> weightslevelnames = null;
		ArrayList<String> weightsfilternames = null;
		ArrayList<ArrayList<String>> filters = new ArrayList<ArrayList<String>>();
		ArrayList<String> crtWeights = null;
		Hashtable<String,ArrayList<ArrayList<String>>> weights = new Hashtable<String,ArrayList<ArrayList<String>>>();
		
		System.out.println("Parsing : "+fname);
		while (curIndex<fname.length()) {
			System.out.println("\n"+fname.charAt(curIndex));
			if(fname.charAt(curIndex)=='{') {
				keyOrValue = 0;
				nestedParentheses++;
			}
			else if(fname.charAt(curIndex)=='[') {
				keyOrValue = 0;
				nestedSQParentheses++;
			}
			else if(fname.charAt(curIndex)=='}') {
				nestedParentheses--;
				if (nestedParentheses == 1 && branch == LEVELS) {
					Config.LayerConfig c = new Config.LayerConfig();
					c.dilation = parseInt(dilation);
					c.stride = parseInt(stride);
					c.Xo = parseInt(Xo);
					c.Yo = parseInt(Yo);
					c.Do = parseInt(Do);
					c.fx = parseInt(Fx);
					c.fy = parseInt(Fy);
					c.px = parseInt(Px);
					c.py = parseInt(Py);
					c.type = CNNtype(type);
					levelsarray.add(c);
					System.out.println("Added: {" + c + ")");
					subbranch = -1;
					branches --;
				} else {
					//System.out.println("nested: " + nestedParentheses +" branch="+branch);
				}
			}
			else if(fname.charAt(curIndex)==']') {
				nestedSQParentheses--;
				
				if (branch == WEIGHTS && branches == 2 && subbranch == 1) {
					branches --;
					weights.put(weightslevelnames.get(weightslevelnames.size()-1), filters);
					subbranch = -1;
				}

				if (branch == WEIGHTS && branches == 3 && subbranch == 2) {
					branches --;
					filters.add(//weightsfilternames.get(weightsfilternames.size()-1), 
							crtWeights);
					subbranch = 1;
				}
			}
			else if(fname.charAt(curIndex)=='"' || fname.charAt(curIndex)=='\'' || Character.isDigit(fname.charAt(curIndex))){
				int closeQuoteIndex = fname.indexOf('"', curIndex+1);
				if (fname.charAt(curIndex)=='\'') closeQuoteIndex = fname.indexOf('\'', curIndex+1);
				String someStringThatIsMaybeKeyMaybeValue;
				if (Character.isDigit(fname.charAt(curIndex))) {
					closeQuoteIndex = -1;
					int _closeQuoteIndex = fname.indexOf(' ', curIndex+1);
					if (_closeQuoteIndex >= 0) closeQuoteIndex = closeQuoteIndex==-1?_closeQuoteIndex:Math.min(closeQuoteIndex, _closeQuoteIndex);
					_closeQuoteIndex = fname.indexOf(',', curIndex+1);
					if (_closeQuoteIndex >= 0) closeQuoteIndex = closeQuoteIndex==-1?_closeQuoteIndex:Math.min(closeQuoteIndex, _closeQuoteIndex);
					_closeQuoteIndex = fname.indexOf('}', curIndex+1);
					if (_closeQuoteIndex >= 0) closeQuoteIndex = closeQuoteIndex==-1?_closeQuoteIndex:Math.min(closeQuoteIndex, _closeQuoteIndex);
					_closeQuoteIndex = fname.indexOf(']', curIndex+1);
					if (_closeQuoteIndex >= 0) closeQuoteIndex = closeQuoteIndex==-1?_closeQuoteIndex:Math.min(closeQuoteIndex, _closeQuoteIndex);
					if (closeQuoteIndex == -1) {
						throw new RuntimeException("No matching end of number");
					}
					someStringThatIsMaybeKeyMaybeValue = fname.substring(curIndex, closeQuoteIndex);
					curIndex=closeQuoteIndex-1;
				} else { 
					//should never happen hopefully
					if(closeQuoteIndex==-1){throw new RuntimeException("No matching quote");}				 
					someStringThatIsMaybeKeyMaybeValue = fname.substring(curIndex+1, closeQuoteIndex);
					curIndex=closeQuoteIndex;
				}
				//System.out.println(someStringThatIsMaybeKeyMaybeValue);

				if (branch == -1) {
					if ("weights".equals(someStringThatIsMaybeKeyMaybeValue)) { 
						weightslevelnames = new ArrayList<String>();
						branches++; branch=WEIGHTS;
					}

					if ("levels".equals(someStringThatIsMaybeKeyMaybeValue)) { 
						levelsarray = new ArrayList<Object>();
						levelnames = new ArrayList<String>(); 
						branches++; branch=LEVELS; 
					}
				} else {
					switch (branch) {
					case WEIGHTS:
						if (subbranch == -1) {
							weightslevelnames.add(someStringThatIsMaybeKeyMaybeValue);
							System.out.println("sublevel: "+someStringThatIsMaybeKeyMaybeValue);
							subbranch = 1;
							weightsfilternames = new ArrayList<String>();
							filters = new ArrayList<ArrayList<String>>();
							branches ++;
						} else {
							if (branches == 2) {
								weightsfilternames.add(someStringThatIsMaybeKeyMaybeValue);
								System.out.println("subsublevel: "+someStringThatIsMaybeKeyMaybeValue);
								subbranch = 2;
								branches ++;
								crtWeights = new ArrayList<String>();
							} else {
								crtWeights.add(someStringThatIsMaybeKeyMaybeValue);
							}
						}
						break;
					case LEVELS: 
						if (subbranch == -1) {
							levelnames.add(someStringThatIsMaybeKeyMaybeValue);
						    System.out.println("sublevel: "+someStringThatIsMaybeKeyMaybeValue);
							subbranch ++;
							branches ++;
						} else {
							if (keyOrValue == 0) {
								if ("xo".equals(someStringThatIsMaybeKeyMaybeValue)) {property = XO;}
								else if ("yo".equals(someStringThatIsMaybeKeyMaybeValue)) {property = YO;}
								else if ("do".equals(someStringThatIsMaybeKeyMaybeValue)) {property = DO;}
								else if ("fx".equals(someStringThatIsMaybeKeyMaybeValue)) {property = FX;}
								else if ("fy".equals(someStringThatIsMaybeKeyMaybeValue)) {property = FY;}
								else if ("px".equals(someStringThatIsMaybeKeyMaybeValue)) {property = PX;}
								else if ("py".equals(someStringThatIsMaybeKeyMaybeValue)) {property = PY;}
								else if ("type".equals(someStringThatIsMaybeKeyMaybeValue)) {property = TYPE;}
								else if ("stride".equals(someStringThatIsMaybeKeyMaybeValue)) {property = STRIDE;}
								else if ("dilation".equals(someStringThatIsMaybeKeyMaybeValue)) {property = DILATION;}
								//System.out.println("(Key)"+someStringThatIsMaybeKeyMaybeValue+" "+property);
							} else {
								//System.out.println("(Property)"+property);
								switch (property) {
								case XO: Xo = someStringThatIsMaybeKeyMaybeValue; break;
								case YO: Yo = someStringThatIsMaybeKeyMaybeValue; break;
								case DO: Do = someStringThatIsMaybeKeyMaybeValue; break;
								case FX: Fx = someStringThatIsMaybeKeyMaybeValue; break;
								case FY: Fy = someStringThatIsMaybeKeyMaybeValue; break;
								case PX: Px = someStringThatIsMaybeKeyMaybeValue; break;
								case PY: Py = someStringThatIsMaybeKeyMaybeValue; break;
								case TYPE: type = someStringThatIsMaybeKeyMaybeValue; break;
								case STRIDE: stride = someStringThatIsMaybeKeyMaybeValue; break;
								case DILATION: dilation = someStringThatIsMaybeKeyMaybeValue; break;
								}
							}
						}
						break;
						
					}
				}
			}
			else if(fname.charAt(curIndex)==',')
			{//nu stiu ce schimba, poate useless
				keyOrValue = 0;
				//System.out.println("Switch to key");
			}
			else if(fname.charAt(curIndex)==':') {
				keyOrValue = 1;
			}
			curIndex++;
		}
		dumpWeights(weights);
	}
	
	public static void loadCNN() {
		String structure =
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
		parseJSONConfig(structure);
		
		//Item i = (Item) parseDeepJSON(structure);
		//dumpDeepJSON(i);
	}
	static void parseJSONConfig(String fname) {
		int curIndex=0;
		int nestedParentheses=0;
		int nestedSQParentheses=0;
		int subbranch = -1, branch = -1, property = -1, branches = 0;
		final int LEVELS = 0;
		final int LEVEL = 1;

		final int XO = 0;
		final int YO = 1;
		final int DO = 2;
		final int FX = 3;
		final int FY = 4;
		final int PX = 5;
		final int PY = 6;
		final int STRIDE = 7;
		final int DILATION = 8;
		final int TYPE = 9;
		String Xo = null, Yo = null, Do = null, 
				Fx = null, Fy = null, 
				Px = null, Py = null,
				stride = null, dilation = null, type = null;

		int keyOrValue=0;//nu stiu daca incepe cu key sau value; 0 key, 1 value
		ArrayList<String> levelnames = null;
		ArrayList<Object> levelsarray = null;

		System.out.println("Parsing : "+fname);
		while (curIndex<fname.length()) {
			//System.out.println("\n"+fname.charAt(curIndex));
			if(fname.charAt(curIndex)=='{') {
				keyOrValue = 0;
				nestedParentheses++;
			}
			else if(fname.charAt(curIndex)=='[') {
				keyOrValue = 0;
				nestedSQParentheses++;
			}
			else if(fname.charAt(curIndex)=='}') {
				nestedParentheses--;
				if (nestedParentheses == 1 && branch == LEVELS) {
					Config.LayerConfig c = new Config.LayerConfig();
					c.dilation = parseInt(dilation);
					c.stride = parseInt(stride);
					c.Xo = parseInt(Xo);
					c.Yo = parseInt(Yo);
					c.Do = parseInt(Do);
					c.fx = parseInt(Fx);
					c.fy = parseInt(Fy);
					c.px = parseInt(Px);
					c.py = parseInt(Py);
					c.type = CNNtype(type);
					levelsarray.add(c);
					System.out.println("Added: {" + c + ")");
					subbranch = -1;
					branches --;
				} else {
					//System.out.println("nested: " + nestedParentheses +" branch="+branch);
				}
			}
			else if(fname.charAt(curIndex)==']') {
				nestedSQParentheses--;
			}
			else if(fname.charAt(curIndex)=='"' || fname.charAt(curIndex)=='\'' || Character.isDigit(fname.charAt(curIndex))){
				int closeQuoteIndex = fname.indexOf('"', curIndex+1);
				if (fname.charAt(curIndex)=='\'') closeQuoteIndex = fname.indexOf('\'', curIndex+1);
				String someStringThatIsMaybeKeyMaybeValue;
				if (Character.isDigit(fname.charAt(curIndex))) {
					closeQuoteIndex = -1;
					int _closeQuoteIndex = fname.indexOf(' ', curIndex+1);
					if (_closeQuoteIndex >= 0) closeQuoteIndex = closeQuoteIndex==-1?_closeQuoteIndex:Math.min(closeQuoteIndex, _closeQuoteIndex);
					_closeQuoteIndex = fname.indexOf(',', curIndex+1);
					if (_closeQuoteIndex >= 0) closeQuoteIndex = closeQuoteIndex==-1?_closeQuoteIndex:Math.min(closeQuoteIndex, _closeQuoteIndex);
					_closeQuoteIndex = fname.indexOf('}', curIndex+1);
					if (_closeQuoteIndex >= 0) closeQuoteIndex = closeQuoteIndex==-1?_closeQuoteIndex:Math.min(closeQuoteIndex, _closeQuoteIndex);
					_closeQuoteIndex = fname.indexOf(']', curIndex+1);
					if (_closeQuoteIndex >= 0) closeQuoteIndex = closeQuoteIndex==-1?_closeQuoteIndex:Math.min(closeQuoteIndex, _closeQuoteIndex);
					if (closeQuoteIndex == -1) {
						throw new RuntimeException("No matching end of number");
					}
					someStringThatIsMaybeKeyMaybeValue = fname.substring(curIndex, closeQuoteIndex);
					curIndex=closeQuoteIndex-1;
				} else { 
					//should never happen hopefully
					if(closeQuoteIndex==-1){throw new RuntimeException("No matching quote");}				 
					someStringThatIsMaybeKeyMaybeValue = fname.substring(curIndex+1, closeQuoteIndex);
					curIndex=closeQuoteIndex;
				}
				//System.out.println(someStringThatIsMaybeKeyMaybeValue);

				if (branch == -1) {
					if ("levels".equals(someStringThatIsMaybeKeyMaybeValue)) { 
						levelsarray = new ArrayList<Object>();
						levelnames = new ArrayList<String>(); 
						branches++; branch=LEVELS; 
						}
				} else {
					switch (branch) {
					case LEVELS: 
						if (subbranch == -1) {
							levelnames.add(someStringThatIsMaybeKeyMaybeValue);
						    System.out.println("sublevel: "+someStringThatIsMaybeKeyMaybeValue);
							subbranch ++;
						} else {
							if (keyOrValue == 0) {
								if ("xo".equals(someStringThatIsMaybeKeyMaybeValue)) {property = XO;}
								else if ("yo".equals(someStringThatIsMaybeKeyMaybeValue)) {property = YO;}
								else if ("do".equals(someStringThatIsMaybeKeyMaybeValue)) {property = DO;}
								else if ("fx".equals(someStringThatIsMaybeKeyMaybeValue)) {property = FX;}
								else if ("fy".equals(someStringThatIsMaybeKeyMaybeValue)) {property = FY;}
								else if ("px".equals(someStringThatIsMaybeKeyMaybeValue)) {property = PX;}
								else if ("py".equals(someStringThatIsMaybeKeyMaybeValue)) {property = PY;}
								else if ("type".equals(someStringThatIsMaybeKeyMaybeValue)) {property = TYPE;}
								else if ("stride".equals(someStringThatIsMaybeKeyMaybeValue)) {property = STRIDE;}
								else if ("dilation".equals(someStringThatIsMaybeKeyMaybeValue)) {property = DILATION;}
								//System.out.println("(Key)"+someStringThatIsMaybeKeyMaybeValue+" "+property);
							} else {
								//System.out.println("(Property)"+property);
								switch (property) {
								case XO: Xo = someStringThatIsMaybeKeyMaybeValue; break;
								case YO: Yo = someStringThatIsMaybeKeyMaybeValue; break;
								case DO: Do = someStringThatIsMaybeKeyMaybeValue; break;
								case FX: Fx = someStringThatIsMaybeKeyMaybeValue; break;
								case FY: Fy = someStringThatIsMaybeKeyMaybeValue; break;
								case PX: Px = someStringThatIsMaybeKeyMaybeValue; break;
								case PY: Py = someStringThatIsMaybeKeyMaybeValue; break;
								case TYPE: type = someStringThatIsMaybeKeyMaybeValue; break;
								case STRIDE: stride = someStringThatIsMaybeKeyMaybeValue; break;
								case DILATION: dilation = someStringThatIsMaybeKeyMaybeValue; break;
								}
							}
						}
					}
				}
			}
			else if(fname.charAt(curIndex)==',')
			{//nu stiu ce schimba, poate useless
				keyOrValue = 0;
				//System.out.println("Switch to key");
			}
			else if(fname.charAt(curIndex)==':') {
				keyOrValue = 1;
			}
			curIndex++;
		}
	}

	private static int parseInt(String s) {
		if (s == null) return -2;
		try {
			return Integer.parseInt(s);
		} catch (Exception e) {e.printStackTrace();}
		return -1;
	}
	private static int CNNtype(String type) {
		if ("INPUT".equals(type)) return Config.INPUT;
		if ("RELU".equals(type)) return Config.RELU;
		if ("SIGMOID".equals(type)) return Config.SIGMOID;
		if ("TANH".equals(type)) return Config.TANH;
		if ("POOLMAX".equals(type)) return Config.POOLMAX;
		if ("POOLAVG".equals(type)) return Config.POOLAVG;
		return -1;
	}
	public static String readFile(String filePath) {
		// String filePath = "doc.txt";
		String content = null;
		try { content = readFile(filePath, StandardCharsets.UTF_8);}
		catch (IOException e) { e.printStackTrace(); }
		//System.out.println(content);
		return content;
	}

	public static String readFile(String path, Charset encoding) throws IOException
	{
		//Files.readString(Path.of("/your/directory/path/file.txt"));
		byte[] encoded = Files.readAllBytes(Paths.get(path));
		return new String(encoded, encoding);
	}

	static void parseJSONConfigFile(String fname) {
		String json = readFile(fname);
		parseJSONConfig (json);
	}

	/*
	void parseJSON(String fname) {
		int curIndex=0;
		int nestedParentheses=0;
		int nestedSQParentheses=0;
		int branch = -1, property = -1;
		final int STATE = 0;
		final int START = 1;
		final int END = 2;
		final int ACTIONS = 3;
		final int TRANSITIONS = 4;

		final int FROM = 0;
		final int TO = 1;
		final int PROB = 2;
		final int ACTION = 3;
		String from = null, to = null, action = null, prob = null;

		int keyOrValue=0;//nu stiu daca incepe cu key sau value; 0 key, 1 value

		//System.out.println("Parsing : "+fname);
		while (curIndex<fname.length()) {
			if(fname.charAt(curIndex)=='{') {
				keyOrValue = 0;
				nestedParentheses++;
			}
			else if(fname.charAt(curIndex)=='[') {
				keyOrValue = 0;
				nestedSQParentheses++;
			}
			else if(fname.charAt(curIndex)=='}') {
				nestedParentheses--;
				if (branch == TRANSITIONS) {
					int t_action = -1, t_from = -1, t_to = -1;
					for (int i = 0; i < actions.size(); i ++)
						if (actions.get(i).equals(action)) t_action = i;
					for (int i = 0; i < state.size(); i ++)
						if (state.get(i).equals(from)) t_from = i;
					for (int i = 0; i < state.size(); i ++)
						if (state.get(i).equals(to)) t_to = i;


					Hashtable<Integer,Transition> crt =
							(Hashtable<Integer,Transition>) transition[t_from];
					if (crt == null) 
						transition[t_from] = crt = new Hashtable<Integer,Transition>();

					Transition t = crt.get(t_action);
					if ( t == null ) {
						crt.put(t_action, t = new Transition());

						t.prob_per_target = new double[state.size()];
						t.action = t_action;
					}
					t.prob_per_target[t_to] = Double.parseDouble(prob);		
					//System.out.println("INPUT: "+from+" "+action+" "+to+" "+prob);
				}
			}
			else if(fname.charAt(curIndex)==']') {
				nestedSQParentheses--;
				branch = -1;
			}
			else if(fname.charAt(curIndex)=='"' || Character.isDigit(fname.charAt(curIndex))){
				int closeQuoteIndex = fname.indexOf('"', curIndex+1);
				String someStringThatIsMaybeKeyMaybeValue;
				if (Character.isDigit(fname.charAt(curIndex))) {
					closeQuoteIndex = fname.indexOf(' ', curIndex+1);
					if(closeQuoteIndex==-1){throw new RuntimeException("No matching quote");}
					someStringThatIsMaybeKeyMaybeValue = fname.substring(curIndex, closeQuoteIndex);
				} else { 
					if(closeQuoteIndex==-1){throw new RuntimeException("No matching quote");}				 
					someStringThatIsMaybeKeyMaybeValue = fname.substring(curIndex+1, closeQuoteIndex);
				}
				//should never happen hopefully
				curIndex=closeQuoteIndex;

				if (branch == -1) {
					if ("states".equals(someStringThatIsMaybeKeyMaybeValue)) { state = new ArrayList<String>(); branch=STATE; }
					else if ("start".equals(someStringThatIsMaybeKeyMaybeValue)) { start = new double[state.size()]; branch=START; }
					else if ("end".equals(someStringThatIsMaybeKeyMaybeValue)) { end = new double[state.size()]; branch=END; }
					else if ("actions".equals(someStringThatIsMaybeKeyMaybeValue)) { actions = new ArrayList<String>(); branch=ACTIONS; }
					else if ("transitions".equals(someStringThatIsMaybeKeyMaybeValue)) {
						transition = new Object[state.size()];
						for (int i = 0; i < state.size(); i ++) {
							transition[i] = new Hashtable<Integer,Transition>();
						}
						branch = TRANSITIONS; 
					}
				} else {
					switch (branch) {
					case STATE: state.add(someStringThatIsMaybeKeyMaybeValue); break;
					case START: 
						for (int i = 0; i < state.size(); i ++ ) 
							if (state.get(i).equals(someStringThatIsMaybeKeyMaybeValue))
							{start[i] = 1.0; break;}
						break;
					case END:
						for (int i = 0; i < state.size(); i ++ ) 
							if (state.get(i).equals(someStringThatIsMaybeKeyMaybeValue))
							{end[i] = 1.0; break;}
						break;
					case ACTIONS: actions.add(someStringThatIsMaybeKeyMaybeValue); break;
					case TRANSITIONS: 
						if (keyOrValue == 0) {
							if ("from".equals(someStringThatIsMaybeKeyMaybeValue)) {property = FROM;}
							else if ("to".equals(someStringThatIsMaybeKeyMaybeValue)) {property = TO;}
							else if ("action".equals(someStringThatIsMaybeKeyMaybeValue)) {property = ACTION;}
							else if ("prob".equals(someStringThatIsMaybeKeyMaybeValue)) {property = PROB;}
						} else {
							switch (property) {
							case FROM: from = someStringThatIsMaybeKeyMaybeValue; break;
							case TO: to = someStringThatIsMaybeKeyMaybeValue; break;
							case ACTION: action = someStringThatIsMaybeKeyMaybeValue; break;
							case PROB: prob = someStringThatIsMaybeKeyMaybeValue; break;
							}
						}
					}
				}
			}
			else if(fname.charAt(curIndex)==',')
			{//nu stiu ce schimba, poate useless
				keyOrValue = 0;
			}
			else if(fname.charAt(curIndex)==':')
				keyOrValue = 1;
			curIndex++;
		}
	}
	*/
}
