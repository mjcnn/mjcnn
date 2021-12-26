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
import java.io.RandomAccessFile;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class JSONTree {
	
	/**
	 * The object for a node in the JSON tree that is not a vector
	 * @author silaghi
	 *
	 */
	public static class Item {
		public Object key;  /* Key if the object has one, otherwise, null*/
		boolean hasKey; /* True if the key has a valid content */
		public Object data; /* The value/content of the item */
		
		public String toString() {
			if (hasKey) return "\n\"" + key.toString() + "\"" + ": " + data.toString();
			
			if (data instanceof String) return "\"" + data + "\"";
			return (data==null)?"":(""+data);
		}
	}
	/**
	 * 
	 * @author silaghi
	 *
	 */
	public static class Node {
		public Object o;                 /* If the node is a single String or Item, it can be stored here */
		public ArrayList<Object> values; /* The values in the vector*/
		public ArrayList<Object> keys;   /* the keys of the items in a vector, can be null*/
		public ArrayList<Item> items;    /* vectors of items*/
		public Hashtable<String,Object> hash; /* if keys are available, items are also placed here. Only last key, on duplicates */
		public char type;               /* Store the type of vector: '{' '['*/
		public String[] stringArray;
		public float[] floatArray;
		public int idxStart  = -1;
		public int idxEnd = -1;

		public String toString() {
			if (o != null) {
				if (o instanceof String) return "\"" + o.toString() + "\"";
				return o.toString();
			}
			String result = "";
			
			if (items != null)
				result = items.stream().map(i -> (""+i)).collect(Collectors.joining(", ", ""+type, (type=='[')?"]":"}"));
			if (floatArray != null)
				result = "\n\t [ floats: " + floatArray.length + " : " + floatArray[0] + "]\n" ;
			if (idxStart >= 0)
			 return "\n\t [ range String : " + idxStart + " - " + idxEnd + "]\n";
			return result;
		}
	}
	private static final boolean DEBUG_JSON = false;
	public static boolean Squares_Delimit_Floats = false;
	static Pattern pattern_arrayend = Pattern.compile("[\\]\\}]");
	static Pattern pattern_wordend = Pattern.compile("[ \\,\\]\\}]");
	static Pattern pattern_cwordend = Pattern.compile("[ \\:\\,\\]\\}]");
	static Pattern pattern_arrayspoiler = Pattern.compile("[\\:\\[\\{]");
	static Matcher macher_arrayend = null;
	static Matcher macher_wordend = null;
	static Matcher macher_cwordend = null;
	static Matcher macher_arrayspoiler = null;
	
	/**
	 * Used to return a parse tree and the next index to be process in the parsed string
	 * @author silaghi
	 *
	 */
	private static class Result {
		Item item = new Item();
		int nextIndex;
	}
	
	static class FileData implements CharSequence {
		String fname;
		String data;
		int start;
		Path path;
		RandomAccessFile raf;
		long file_length;
		final int BUFFER_INCREMENT = 1024*1024;
		final int BUFFER_MARGE = 1024;
		byte[] encoded = new byte[BUFFER_INCREMENT];
		int encoded_len;
		
		FileData (String fname) {
			this.fname = fname;
			path = Paths.get(fname);
			try {
				file_length = Files.size(path);
				
				raf = new RandomAccessFile(fname, "r");
				load(0);
				//encoded = Files.readAllBytes(path);
			} catch (IOException e) {
				data = "";
				start = 0; file_length = 0;
				e.printStackTrace();
			}
		}
		
		FileData (String fname, String txt) {
			this.data = txt;
			file_length = encoded_len = data.length();
			start = 0;
		}
		
		public char charAt(int idx) {
			if ( ! has(idx) )
				load(Math.max(0, idx - BUFFER_MARGE));
			
			return data.charAt(idx - this.start);
		}
		
		String substring(int start, int end) {
			try {
			if (start - this.start >=0 && end - this.start < encoded_len)
				return data.substring(start - this.start, end - this.start);
			} catch (Exception e) {
				System.out.println(" this.start="+this.start+" start="+start+" end="+end+" elen="+encoded_len+" dlen="+data.length());
				e.printStackTrace(); throw e;
			}
			
			if (end-start > encoded.length) encoded = new byte[((end - start)/BUFFER_INCREMENT+1)*BUFFER_INCREMENT];
			
			load(start);
			return data.substring(start - this.start, end - this.start);
		}
		
		@Override
		public int length() {
			return (int)file_length;
		}
		@Override
		public CharSequence subSequence(int start, int end) {
			return substring(start, end);
		}
		boolean has (int idx) {
			if (idx >= start && idx < start + encoded_len) return true;
			return false;
		}
		int readMax(RandomAccessFile raf, byte[] buf) {
			int result = 0;
			try {
				while (result < buf.length) {
					int crt = raf.read(buf, result, buf.length - result);
					if (crt <= 0) break;
					result += crt;
				}
			} catch (Exception e) {e.printStackTrace();}
			return result;
		}
		void load (int idxFrom) {
			try {
				raf.seek(this.start = idxFrom);
				encoded_len = readMax(raf,encoded);
				data = new String(encoded, 0, encoded_len, StandardCharsets.US_ASCII);
			} catch (Exception e) {e.printStackTrace();}			
		}
		public int indexOf(char ch, int idxFrom) {
			int result;
			if (! has(idxFrom)) load (Math.max(0, idxFrom - BUFFER_MARGE));
				
			result = data.indexOf(ch, idxFrom - start); 
			while (result == -1 && start + encoded_len < file_length) {
				load( start + encoded_len );
				result = data.indexOf(ch, 0); 
			}
			if (result >= 0) result += this.start;
			return result;
		}
	}
	
	public static Item parseDeepJSONFile(String fname) {
		FileData data = new FileData(fname); 
		
		int crtIndex = 0;
		macher_arrayend = pattern_arrayend.matcher(data);
		macher_wordend = pattern_wordend.matcher(data);
		macher_cwordend = pattern_cwordend.matcher(data);
		macher_arrayspoiler = pattern_arrayspoiler.matcher(data);
		Result r = parseDeepJSON(data, crtIndex, 1);
		return r.item;
		
	}
		
	public static Item parseDeepJSON(String idata) {
		FileData data = new FileData(null, idata); 
		int crtIndex = 0;
		macher_arrayend = pattern_arrayend.matcher(data);
		macher_wordend = pattern_wordend.matcher(data);
		macher_cwordend = pattern_cwordend.matcher(data);
		macher_arrayspoiler = pattern_arrayspoiler.matcher(data);
		Result r = parseDeepJSON(data, crtIndex, 1);
		return r.item;
	}
	
	public static void dumpDeepJSON(Object tree) {
		if (tree instanceof String) {
			System.out.print("'" +tree+"'");
		}
		if (tree instanceof Float) {
			System.out.print("" +tree);
		}
		else if (tree instanceof Item) {
			Item i = (Item) tree;
			if (i.key != null) System.out.print("" + i.key + ":");
			dumpDeepJSON(i.data);
		}
		else if (tree instanceof Node) {
			Node n = (Node)tree;
			System.out.print(" "+n.type+" ");
			if ( n.items != null ) {
				for (Item i: n.items) {
					if (i.key != null) System.out.print("\n" + i.key + ":");
					dumpDeepJSON(i.data);
					System.out.print(" , ");
				}
			}
			if (n.floatArray != null) {
				for (float i : n.floatArray) System.out.print("" + i + ", ");
			}
			if (n.stringArray != null) {
				for (String i : n.stringArray) System.out.print("" + i + ", ");
			}
			System.out.print(" "+((n.type=='[')?']':'}')+" ");
		}
		else System.out.print(tree);
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
    public static int countOccurence(FileData data, char ch, int start, int end){
        int count = 0;
        for ( int i = start; i < end ; i ++ ) {
            if ( data.charAt(i) == ch ) count++;
        }
        return count;
     }
	public static Result parseDeepJSON(FileData data, int curIndex, int rec) {
		JSONTree.Node n = new Node();
		Result result = new Result();
		boolean vector = false;
		//System.out.println("Recursive rec "+rec+": "+data.substring(curIndex));
		while (curIndex < data.length()) {
			char _x_ = data.charAt(curIndex);
			// most likely match first
			// if (Character.isWhitespace(_x_)) { curIndex ++; continue; }
			
			if (_x_ == ' ') { curIndex ++; continue; }
			
			if ( _x_ == ',' ) {
				// System.out.println(",*"+curIndex);
				if ( ! vector ) {
					result.nextIndex = curIndex;
					result.item.data = result.item.key;
					result.item.key = null;
					//System.out.println("Returning recursive value, ");
					return result;	
				}
				
				Result r = parseDeepJSON(data, curIndex + 1, rec+1);
				if (r == null) throw new RuntimeException("Premature JSON end in block: "+data.substring(curIndex+1, curIndex+10));
				//System.out.println("Done rec:"+rec+" "+data.substring(curIndex+1) +" \n->" + data.substring(r.nextIndex));

				n.values.add(r.item.data);
				n.keys.add(r.item.key);
				n.items.add(r.item);
				if (r.item.hasKey) n.hash.put(r.item.key.toString(), r.item.data);
				curIndex = r.nextIndex;
				continue;				
			}

			if ( 
					Character.isDigit(_x_) || (_x_ == '-')
					|| _x_=='"' 
					|| _x_=='\'' 
					|| Character.isLetter(_x_)
					) {
				String someStringThatIsMaybeKeyMaybeValue;
				int closeQuoteIndex;
				if (Character.isDigit(_x_) || (_x_ == '-')) {
					closeQuoteIndex = -1;

					if (! macher_wordend.find( curIndex + 1 )) throw new RuntimeException("No matching number end at "+curIndex);
					closeQuoteIndex = macher_wordend.start();

					someStringThatIsMaybeKeyMaybeValue = data.substring(curIndex, closeQuoteIndex);
					curIndex=closeQuoteIndex-1;
					result.item.key = Float.parseFloat(someStringThatIsMaybeKeyMaybeValue);
				} else
				if (Character.isLetter(_x_)) {					
					if (! macher_cwordend.find( curIndex + 1 )) throw new RuntimeException("No matching letter end at "+curIndex);
					closeQuoteIndex = macher_cwordend.start();

					someStringThatIsMaybeKeyMaybeValue = data.substring(curIndex, closeQuoteIndex);
					curIndex=closeQuoteIndex-1;
					result.item.key = someStringThatIsMaybeKeyMaybeValue;
				} else { 
					/*  Either " or ' */
					if (_x_=='\'') closeQuoteIndex = data.indexOf('\'', curIndex+1);
					else closeQuoteIndex = data.indexOf('"', curIndex+1);
					
					//should never happen hopefully
					if (closeQuoteIndex == -1) {throw new RuntimeException("No matching quote");}				 
					someStringThatIsMaybeKeyMaybeValue = data.substring(curIndex+1, closeQuoteIndex);
					curIndex=closeQuoteIndex;
					result.item.key = someStringThatIsMaybeKeyMaybeValue;
				}
				
				curIndex ++;
				continue;
			}

			if ( _x_ == ':' ) {
				//System.out.println("Key :  " + result.item.key);
				
				//keyNotValue = false;
				Result r = parseDeepJSON(data, curIndex+1, rec +1);
				if (r == null) throw new RuntimeException("Premature JSON end in value: "+data.substring(curIndex+1, curIndex+10));
				//System.out.println("Done rec:"+rec+" "+data.substring(curIndex+1) +" \n->" + data.substring(r.nextIndex));
				result.item.data = r.item.data;
				result.nextIndex = r.nextIndex;
				result.item.hasKey =  true;

				return result;
			}
			
			if ( _x_ == '[' && Squares_Delimit_Floats) {
				vector = true;
				n.type = _x_;
				int closeQuoteIndex = -1;
				
				if (! macher_arrayend.find( curIndex + 1 )) throw new RuntimeException("No matching paranthesis at "+curIndex);
				closeQuoteIndex = macher_arrayend.start();
				
				if (DEBUG_JSON) {
					macher_arrayspoiler.region(curIndex, closeQuoteIndex);
					if (macher_arrayspoiler.matches()) throw new RuntimeException("Spoiled array "+curIndex+" at "+macher_arrayspoiler.start());
				}
				
				int countCommas = countOccurence(data, ',', curIndex+1, closeQuoteIndex);
				n.floatArray = new float[countCommas + 1];
				int i = curIndex + 1, nextBeginning = curIndex + 1, k = 0;
				try {
					for (; i < closeQuoteIndex; i ++) {
						if (data.charAt(i) == ',') {
							n.floatArray[k ++] = Float.parseFloat(data.substring(nextBeginning, i));
							nextBeginning = i + 1;
						}
					}
					n.floatArray[k] = Float.parseFloat(data.substring(nextBeginning, i));
				}
				catch (java.lang.NumberFormatException e) {
					// System.out.println( e.getLocalizedMessage() );
					n.floatArray = null;
					n.idxStart = curIndex+1;
					n.idxEnd = closeQuoteIndex;
					n.stringArray = data.substring(curIndex+1, closeQuoteIndex).split(",");
				}
				curIndex = closeQuoteIndex;
				continue;
				
			}

			if ( _x_ == '{' || _x_ == '[' ) {
				vector = true;
				n.type = _x_;
				n.values = new ArrayList<Object>();
				n.keys = new ArrayList<Object>();
				n.items = new ArrayList<Item>();
				n.hash = new Hashtable<String,Object>();

				Result r = parseDeepJSON (data, curIndex + 1, rec +1);
				if (r == null) throw new RuntimeException("Premature JSON end in block: "+data.substring(curIndex+1, curIndex+10));
				//System.out.println("Done rec:"+rec+" "+data.substring(curIndex+1) +" \n->" + data.substring(r.nextIndex));

				n.values.add(r.item.data);
				n.keys.add(r.item.key);
				n.items.add(r.item);
				if (r.item.hasKey) n.hash.put(r.item.key.toString(), r.item.data);
				//else System.out.println("No key for "+r.item.key);
				curIndex = r.nextIndex;
				continue;
			}

			if ( _x_ == '}' || _x_ == ']' ) {
				if ( ! vector ) {
					result.nextIndex = curIndex;
					result.item.data = result.item.key;
					result.item.key = null;
					//System.out.println("Returning recursive value} ");
					return result;	
				}

				Item i = new Item();
				i.hasKey = false;
				i.key = null;
				i.data = n;

				
				result.item = i;
				result.nextIndex = curIndex + 1;
				//System.out.println("Returning recursive vector");
				return result;
			} 
			
			curIndex++;
		}
		return null;
	}

}
