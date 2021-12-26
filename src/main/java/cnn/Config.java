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

import java.util.ArrayList;

import json.JSONTree;
import json.JSONTree.Item;
import json.JSONTree.Node;

public class Config {
    public static final int INPUT = 8;
    public static final int RELU = 1;
    public static final int SIGMOID = 2;
    public static final int TANH = 3;
    public static final int POOLMAX = 4;
    public static final int POOLAVG = 5;
    public static final int SOFTMAX = 6;
    public static final int CONVOLUTION = 7;
    
    public static final boolean FLAT_ARRAYS = true;
    public static final boolean FLAT_INDEXES = true;
    public static final boolean FLAT_INDEXES_POOL = true; // pool perceptron use indexes for level 0
   
	private static int CNNtype(String type) {
		if ("INPUT".equals(type)) return Config.INPUT;
		if ("RELU".equals(type)) return Config.RELU;
		if ("SIGMOID".equals(type)) return Config.SIGMOID;
		if ("TANH".equals(type)) return Config.TANH;
		if ("POOLMAX".equals(type)) return Config.POOLMAX;
		if ("POOLAVG".equals(type)) return Config.POOLAVG;
		if ("SOFTMAX".equals(type)) return Config.SOFTMAX;
		if ("CONVOLUTION".equals(type)) return Config.CONVOLUTION;
		return -1;
	}
	public static String CNNtype(int type) {
		if (Config.INPUT  ==(type)) return "INPUT";
		if (Config.RELU   ==(type)) return "RELU";
		if (Config.SIGMOID==(type)) return "SIGMOID";
		if (Config.TANH   ==(type)) return "TANH";
		if (Config.POOLMAX==(type)) return "POOLMAX";
		if (Config.POOLAVG==(type)) return "POOLAVG";
		if (Config.SOFTMAX==(type)) return "SOFTMAX";
		if (Config.CONVOLUTION==(type)) return "CONVOLUTION";
		return ""+type;
	}
    
    public static class LayerConfig {
    	public int type, Xo, Yo, Do=1, fx=1, fy=1, px, py, stride=1, dilation=0;
		public String name;
    	public String toString() {
    		return "type:"+CNNtype(type)+"\t,Xo:"+Xo+",Yo:"+Yo+",Do:"+Do+",fx:"+fx+",fy:"+fy+",px:"+px+",py:"+py+",s:"+stride+",d:"+dilation;
    	}
    }

    /**
     * Parsing JSON files obtained from KERAS proto files of VGG16 with<p><pre>
     * </pre>
     * @param structure
     * @param fname
     * @return
     */
    // cat vgg16_proto.txt | sed 's/param {/param: {/g; s/[^{]$/&,/g; s/layers {/"layers": {/g; s/^[[:space:]]*//g; s/^\([_a-zA-Z0-9]*\):/"\1":/g; s/: \([_a-zA-Z][_a-zA-Z0-9]*\)/: "\1"/g' | tr "\r\n" "##" | sed 's/,#\([[:space:]]*}\)/\n\1/g' | tr "#" "\n" > vgg16_proto.json
	public static LayerConfig[] parseConfigVGG(String structure, String fname) {
		if (structure == null && fname != null) structure = JSONTree.readFile(fname);
		String top = "data", name = null; // bottom, 
		int prev_X_size, prev_Y_size, prev_pad=0, kernel_size=3, prev_stride=1;//, prev_Depth, prev_dilation=0; 
		int next_Depth = 1; // next_X_size, next_Y_size, 
		
		Item c = JSONTree.parseDeepJSON(structure);		
		JSONTree.Node n = (Node) c.data;
		
		int dimensions[] = new int[4];
		int i = 0;
		for (Item item : n.items) {
			if ("input_dim".equals(item.key)) dimensions[i ++] = ((Float)item.data).intValue();
		}
		//prev_Depth = dimensions[1]; 
		prev_X_size = dimensions[2]; prev_Y_size = dimensions[3];
		System.out.println("D: " + dimensions[0] + "," + dimensions[1]+ "," + dimensions[2]+ "," + dimensions[3]);
		
		ArrayList<LayerConfig> lc = new ArrayList<LayerConfig>();
		
		LayerConfig input = new LayerConfig();
		input.type = Config.INPUT;
		input.Do = dimensions[1];
		input.Xo = dimensions[2];
		input.Yo = dimensions[3];
		lc.add(input);
		
		for (Item item : n.items) {
			if ("layers".equals(item.key)) {
				Node layerNode = (Node) item.data;
				if ("CONVOLUTION".equals(layerNode.hash.get("type"))) {
					String new_bottom = (String) layerNode.hash.get("bottom");
					assert (new_bottom.equals(top));
					//bottom = new_bottom;
					top = (String) layerNode.hash.get("top");
					name = (String) layerNode.hash.get("name");
					Node param = (Node) layerNode.hash.get("convolution_param");
					//prev_Depth = next_Depth;
					next_Depth = ((Float)param.hash.get("num_output")).intValue();
					prev_pad = ((Float)param.hash.get("pad")).intValue();
					kernel_size = ((Float)param.hash.get("kernel_size")).intValue();
					prev_X_size = prev_X_size + 2 * prev_pad - kernel_size + 1;
					prev_Y_size = prev_Y_size + 2 * prev_pad - kernel_size + 1;
				}
				if ("RELU".equals(layerNode.hash.get("type"))) {
					String new_bottom = (String) layerNode.hash.get("bottom");
					assert (new_bottom.equals(top));
					//bottom = new_bottom;
					top = (String) layerNode.hash.get("top");
					String relu_name = (String) layerNode.hash.get("name");
					System.out.println("RELU: " + relu_name);
					LayerConfig relu = new LayerConfig();
					relu.dilation = 0;
					relu.stride = 1;
					relu.type = Config.RELU;
					relu.px = relu.py = prev_pad;
					relu.fx = relu.fy = kernel_size;
					relu.Do = next_Depth;
					relu.Xo = prev_X_size;
					relu.Yo = prev_Y_size;
					relu.name = name + "/" + relu_name;
					lc.add(relu);
				}
				if ("POOLING".equals(layerNode.hash.get("type"))) {
					String new_bottom = (String) layerNode.hash.get("bottom");
					assert (new_bottom.equals(top));
					//bottom = new_bottom;
					top = (String) layerNode.hash.get("top");
					Node param = (Node) layerNode.hash.get("pooling_param");
					prev_stride = ((Float)param.hash.get("stride")).intValue();
					kernel_size = ((Float)param.hash.get("kernel_size")).intValue();
					//String op = ((String)param.hash.get("pool"));
					String pool_name = (String) layerNode.hash.get("name");
					System.out.println("POOL: " + pool_name);
					LayerConfig pool = new LayerConfig();
					pool.dilation = 0;
					pool.stride = prev_stride;
					pool.type = Config.POOLMAX;
					pool.px = pool.py = 0; // prev_pad;
					pool.fx = pool.fy = kernel_size;
					pool.Do = next_Depth;
					pool.Xo = prev_X_size = prev_X_size/prev_stride;
					pool.Yo = prev_Y_size = prev_Y_size/prev_stride;
					pool.name = name + "/" + pool_name;
					lc.add(pool);
					
				}
				if ("INNER_PRODUCT".equals(layerNode.hash.get("type"))) {
					String new_bottom = (String) layerNode.hash.get("bottom");
					assert (new_bottom.equals(top));
					//bottom = new_bottom;
					top = (String) layerNode.hash.get("top");
					String ip_name = (String) layerNode.hash.get("name");
					Node param = (Node) layerNode.hash.get("inner_product_param");
					//prev_Depth = next_Depth;
					next_Depth = ((Float)param.hash.get("num_output")).intValue();
					//next_X_size = next_Y_size = 1;
					System.out.println("INNERPROD: " + ip_name);
					name = ip_name;
					//prev_dilation = 0;
					prev_stride = 1;
					prev_pad = 0;
					kernel_size = prev_X_size;
					prev_X_size = prev_Y_size = 1;
					/*
					LayerConfig ip = new LayerConfig();
					ip.dilation = 0;
					ip.stride = 1;
					ip.Do = next_Depth;
					ip.Xo=ip.Yo=1;
					ip.px = ip.py = 0;
					ip.fx = ip.fy = prev_Depth;
					//lc.add(ip);
					*/
				}
				if ("SOFTMAX".equals(layerNode.hash.get("type"))) {
					String new_bottom = (String) layerNode.hash.get("bottom");
					assert (new_bottom.equals(top));
					//bottom = new_bottom;
					top = (String) layerNode.hash.get("top");
					String softmax_name = (String) layerNode.hash.get("name");
					System.out.println("SOFTMAX: " + softmax_name);
					
					LayerConfig convolution = new LayerConfig();
					convolution.dilation = 0;
					convolution.stride = 1;
					convolution.px = convolution.py = prev_pad;
					convolution.fx = convolution.fy = kernel_size;
					convolution.Do = next_Depth;
					convolution.Xo = prev_X_size;
					convolution.Yo = prev_Y_size;
					convolution.name = name + "/" + softmax_name;
					convolution.type = Config.CONVOLUTION;
					lc.add(convolution);
					
					LayerConfig softmax = new LayerConfig();
					softmax.dilation = 0;
					softmax.stride = 1;
					softmax.px = softmax.py = prev_pad;
					softmax.fx = softmax.fy = kernel_size;
					softmax.Do = next_Depth;
					softmax.Xo = prev_X_size;
					softmax.Yo = prev_Y_size;
					softmax.name = name + "/" + softmax_name;
					softmax.type = Config.SOFTMAX;
					lc.add(softmax);
				}
			}
		}
		//System.out.println(lc.stream().map(x -> x.toString()).collect(Collectors.joining("\n", "\n[\n", "\n]\n")));
		
		LayerConfig[] result = lc.toArray(new LayerConfig[0]);
		return result;
	}
    
    /*
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
*/
	public static LayerConfig[] parseConfig(String structure, String fname) {
		if (structure == null && fname != null) structure = JSONTree.readFile(fname);
		
		Item c = JSONTree.parseDeepJSON(structure);		
		//System.out.println("Root key="+c.key);
		
		//JSONTree.dumpDeepJSON(c);
		//System.out.println(c.getClass());
		JSONTree.Node n = (Node) c.data;
		
		//System.out.println(n.getClass());
		//for (Object o : n.hash.keySet()) System.out.println("HASH "+o+":"+n.hash.get(o));
		//for (Item o : n.items)	System.out.println("ARRAY: "+o.key+":"+o.data);
		//for (Item ni: n.items) System.out.println("NI key="+ni.key);
				
		JSONTree.Node n2 = (Node) n.hash.get("levels");
		//System.out.println(""+n2.type+"n2 levels-> i="+n2.items+" v="+n2.values+" k="+n2.keys+" o="+n2.o+" fA="+n2.floatArray+" fS="+n2.stringArray);
		LayerConfig result[] = new LayerConfig[n2.items.size()];
		
		for (int k = 0; k < result.length; k++) {
			Item i = n2.items.get(k);
			result[k] = new LayerConfig();
			Node m = (Node)i.data;
			for (String j: m.hash.keySet()) {
				  if ("type".equals(j)) result[k].type = CNNtype((String) m.hash.get(j));
				  if ("xo".equals(j)) result[k].Xo = ((Float) m.hash.get(j)).intValue();
				  if ("yo".equals(j)) result[k].Yo = ((Float) m.hash.get(j)).intValue();
				  if ("do".equals(j)) result[k].Do = ((Float) m.hash.get(j)).intValue();
				  if ("fx".equals(j)) result[k].fx = ((Float) m.hash.get(j)).intValue();
				  if ("fy".equals(j)) result[k].fy = ((Float) m.hash.get(j)).intValue();
				  if ("px".equals(j)) result[k].px = ((Float) m.hash.get(j)).intValue();
				  if ("py".equals(j)) result[k].py = ((Float) m.hash.get(j)).intValue();
				  if ("stride".equals(j)) result[k].stride = ((Float) m.hash.get(j)).intValue();
				  if ("dilation".equals(j)) result[k].dilation = ((Float) m.hash.get(j)).intValue();
			}
		}
		//System.out.println(Arrays.asList(result).stream().map(x -> x.toString()).collect(Collectors.joining("\n", "\n[\n", "\n]\n")));
		
		return result;
	}
	
}
