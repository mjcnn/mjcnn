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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import cnn.Config.LayerConfig;
import cnn.util.Array2DF;
import cnn.util.Array2DFimp;
import cnn.util.Array3DF;
import json.JSONTree;
import json.JSONTree.Item;
import json.JSONTree.Node;

public class CNN {
	Array2DF input;      // [d][w x h]
	Array2DF output;     // [d][w x h]
	
	Array2DF hidden[];   // [l][d][w x h]

	Array2DF biases []; // level.depth.1
	Array3DF weights []; // level.o_depth.i_depth.idx

	LayerStub layer[]; // may make layer/filter/biases/weights/hidden an ArrayList to enable easy manipulation
	Filter[][] filter;
	int layersNb;

	/**
	 * Loading a "native" json file with weights as here <p><pre>
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
+ "}";	 </pre>
	 * 
	 * @param parameters
	 * @param fname
	 */
	
	public void loadWeights(String parameters, String fname) {
		if (parameters == null && fname != null) 
			parameters = JSONTree.readFile(fname);
		
		if (parameters == null) return;
		
		Item c = JSONTree.parseDeepJSON(parameters);		
		Node n = (Node) c.data;
		Node n2 = (Node) n.hash.get("weights");
		for (int k = 0; k < n2.values.size(); k ++) {
			if (filter[k].length < 1 || filter[k][0].getWeightsNb() == 0) continue;
			
			Node n3 = (Node) n2.values.get(k);
			for (int f = 0; f < n3.values.size(); f ++) {
				Node m = (Node) n3.values.get(f);
				int i = 0;
				filter[k][f].setBias( (Float)m.values.get(i));
				for (int w = 0; w < filter[k][f].getWeightsNb(); w ++) {
					for (int d = 0; d < filter[k][f].getDepth(); d ++) {
						filter[k][f].setWeight(d, w, (Float)m.values.get( ++i));
					}
				}
			}
		}
		
	}
	/**
	 * Loading JSON files obtained from h5 Keras VGG16 ascii dumps with: <p>
	 * cat vgg16_weights.txt| egrep -v '(STRSIZE|STRPAD|CSET|CTYPE)' | sed 's/^.*)://g;s/DATA \(.*\){/"DATA": [/g;s/DATASET \(.*\) {/"DATASET": { \1\,/g;s/GROUP \(.*\) {/"GROUP": { \1\,/g;s/ATTRIBUTE \(.*\) {/"ATTRIBUTE": { \1\,/g;s/HDF5 \(.*\) {/"HDF5": { \1\,/g;s/\}/\},/g; s/DATATYPE.*{/{/g;s/DATASPACE.*{.*(\(.*\)).*(\(.*\)).*$/"DATASPACE": { [\1], [\2] },/g' | egrep -v 'DATATYPE|DATASPACE ' >vgg16_weights_sq2.json

	 * 
	 * @param parameters
	 * @param fname
	 */
	public void loadWeightsVGG(String parameters, String fname) {
		JSONTree.Squares_Delimit_Floats = true;
		int crt_layer = -1;
		Item c = null;
		if (parameters == null && fname != null) {
			//parameters = JSONTree.readFile(fname);
			c = JSONTree.parseDeepJSONFile(fname);
		} else {
			if (parameters == null) return;
			c = JSONTree.parseDeepJSON(parameters);
		}
		
		Node n_hdf5_content = (Node) c.data;
		//Node n_hdf5_content = (Node) n.hash.get("weights");
		for (Item i : n_hdf5_content.items) {
			if ("GROUP".equals(i.key)) {
				//System.out.println("base group");
				Node n_hdf5_group_content = (Node) i.data;
				for (Item j : n_hdf5_group_content.items) {
					if ("GROUP".equals(j.key)) {
						Node n_hdf5_group_grup_content = (Node) j.data;
						String block_name = (String) n_hdf5_group_grup_content.values.get(0);
						// * System.out.println("group: " + block_name);
						for (Item k: n_hdf5_group_grup_content.items) {
							if ("GROUP".equals(k.key)) {
								Node n_hdf5_group_grup_group_content = (Node) k.data;
								// * String subblock_name = (String) n_hdf5_group_grup_group_content.values.get(0);
								// * System.out.println("group+ " + subblock_name);
								
								Filter[] crt_fil = null;
								while (crt_fil == null || crt_fil[0].getWeightAsVector(crt_layer) == null) {
									crt_fil = filter[++ crt_layer];
								}
								
								// * System.out.println("Layers: "+ crt_fil.length + " - " + layer[crt_layer].name +" "+layer[crt_layer].type);
								
								for (Item l: n_hdf5_group_grup_group_content.items) {
									//System.out.println(" -- " + l.key +":" +l.data);
									if ("DATASET".equals(l.key)) {
										Node n_hdf5_group_grup_group_dataset_content = (Node) l.data;
										String type = (String) n_hdf5_group_grup_group_dataset_content.values.get(0);
										
										assert("DATASPACE".equals(n_hdf5_group_grup_group_dataset_content.values.get(1)));
										Node dataspace = (Node) n_hdf5_group_grup_group_dataset_content.values.get(1);

										Node dimspace = (Node) dataspace.values.get(1);
										// * for (float f: dimspace.floatArray)  System.out.println("  "+f);

										assert (dimspace.floatArray[dimspace.floatArray.length - 1] == layer[crt_layer].getDepth_output());
										//crt_fil.length);
										assert (dimspace.floatArray[dimspace.floatArray.length - 2] == layer[crt_layer].getDepth_input());
										// crt_fil[0].getDepth());
										
										assert("DATA".equals(n_hdf5_group_grup_group_dataset_content.values.get(2)));
										Node data = (Node) n_hdf5_group_grup_group_dataset_content.values.get(2);

									
										
										if ("bias:0".equals(type)) {
											// * System.out.println(" biases: " + data.floatArray.length);
											
											for (int idx = 0; idx < data.floatArray.length; idx ++) {
												crt_fil[idx].setBias(data.floatArray[idx]);
											}
										}
										if ("kernel:0".equals(type)) {
											// * System.out.println(" kernels : " + data.floatArray.length);
											int src = 0;
											int dim = crt_fil[0].getX() * crt_fil[0].getY() * crt_fil[0].getDepth() * crt_fil.length;
											// * System.out.println("filter sz=" + crt_fil[0].getX() +","+ crt_fil[0].getY() +","+ crt_fil[0].getDepth() +","+ crt_fil.length);
											assert (dim == data.floatArray.length);
											
											// TODO: Have to check what is first: X or Y!!!!
											// TODO: could be guessed by classifying an image and its transpose
											
											if (Config.H5_FILTERS_DIMS==Config.H5_FILTERS_X_Y_I_O) {
												for (int i1 = 0; i1 < crt_fil[0].getX(); i1 ++) {
													for (int i2 = 0; i2 < crt_fil[0].getY(); i2 ++) {
														for (int i3 = 0; i3 < layer[crt_layer].getDepth_input(); i3 ++) {
															for (int i4 = 0; i4 < layer[crt_layer].getDepth_output(); i4 ++) {
																crt_fil[i4].setWeight(i1, i2, i3, data.floatArray[src++]);
															}													
														}													
													}
												}											
											}
											
											if (Config.H5_FILTERS_DIMS==Config.H5_FILTERS_Y_X_I_O) {
												for (int i1 = 0; i1 < crt_fil[0].getX(); i1 ++) {
													for (int i2 = 0; i2 < crt_fil[0].getY(); i2 ++) {
														for (int i3 = 0; i3 < layer[crt_layer].getDepth_input(); i3 ++) {
															for (int i4 = 0; i4 < layer[crt_layer].getDepth_output(); i4 ++) {
																crt_fil[i4].setWeight(i1, i2, i3, data.floatArray[src++]);
															}													
														}													
													}
												}
											}
											
										}
									}
								}
							}
						}
					}
				}
			} // else System.out.println("base other: " + i.key + " " + i.data);
		}
		set_weights_in_GPU_kernels();
	}	
	/**
	 * Procedure to convert a CNN to a native JSON string in given file to be created.
	 * @param fname
	 * @return
	 */
	public String storeWeightsNative (String fname) {
		String s = "{'weights':[";
		for (int k = 0; k < filter.length; k ++) {
			if (k > 0) s+= ", ";
			s += "\n'layer"+k+"':[";
			for (int f = 0; f < filter[k].length; f ++) {
				if (f > 0) s+= ", ";
				s += "\n\t'filter"+k+"_"+f+"':"+
				//filter[k][f].x_size+"x"+filter[k][f].y_size+"="+filter[k][f].getWeightsNb()+"*"+filter[k][f].getDepth()+
				"["; //+"\n\t\t";
				
				//for (int d = 0; d < filter[k][f].getDepth(); d ++) {
					s += filter[k][f].getBias() + "";
				//}
				s += "\n\t\t";
				for (int w = 0; w < filter[k][f].getWeightsNb(); w ++) {
					s += " ";
					for (int d = 0; d < filter[k][f].getDepth(); d ++) {
						s += ", ";
						s += filter[k][f].getWeight(d, w);
					}
					if ((w+1) % filter[k][0].x_size == 0) s += "\n\t\t";
				}
				s += "]";
			}
			s += "]";
		}
		s += "]}";
		
		if (fname != null) {
			try {
				Files.write( Paths.get(fname), s.getBytes());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return s;
	}
	
	/**
	 * The input is supposed to consist of several layers (e.g. RGB).
	 * The first dimension of the array is the layer, the second is the map addressable with Field.getIndex(x,X,y,Y)
	 * @param input
	 */
	public void setInput(Array2DF input) {
		this.input = input;
		layer[0].setInput(input);
	}
	public Array2DF classify_with_CPU() {
		for (int l = 0; l < layer.length; l ++) {
			System.out.println("CPU Level="+l);
			layer[l].convolute_with_CPU();
		}
		return output;
	}

	public Array2DF classify_with_GPU() {
		for (int l = 0; l < layer.length; l ++)
			layer[l].convolute_with_GPU();
		return output;
	}
	
	public void set_weights_in_GPU_kernels() {
		for (LayerStub l: layer) {
			if (l.usesGPU()) l.update_GPU_weights();
		}
	}
	
	public CNN (LayerConfig[] config, Array2DF input, Array2DF output) {
		int h = 0;
		this.input = input;
		this.output = output;
		
		layer = new LayerStub[config.length - 1]; // the perceptrons, no perceptrons for input
		filter = new Filter[layer.length][];
		
		int hiddens = config.length;
		hidden = new Array2DF[hiddens]; // the hidden nodes values (other than input and output)
		hidden[h] = input;
		for (h = 1; h < hiddens-1; h ++) 
			hidden[h] = new Array2DFimp (config[h].Do, config[h].Xo * config[h].Yo);
		hidden[h] = output;
		
		/**
		 * Fully created in filter!
		 */
		biases = new Array2DF[hiddens-1];
		weights = new Array3DF[hiddens-1];

		
		for (int l = 0; l < hiddens-1; l ++) {
			//System.out.println("Create Layer i="+config[l]);
			//System.out.println("Create Layer o="+config[l+1]);
			
			if (config[l+1].type == Config.POOLAVG ||  config[l+1].type == Config.POOLMAX || config[l+1].type == Config.SOFTMAX)
			    filter[l] = Filter.getArrayPoolFilter(config[l+1].Do, config[l+1].fx, config[l+1].fy, config[l].Do);
			else
				filter[l] = Filter.getArrayFilter(
						config[l+1].Do, config[l+1].fx, config[l+1].fy, config[l].Do,
						l, biases, weights);
			
			
			layer[l] =
					new LayerStub(hidden[l], config[l].Xo, config[l].Yo, config[l].Do,
							hidden[l+1], config[l+1].Xo, config[l+1].Yo, config[l+1].Do,
							config[l+1].px, config[l+1].py,
							config[l+1].stride, config[l+1].dilation,
							filter[l], biases[l], weights[l], config[l+1].type, config[l+1].name
							);
		}
		
		
	}
/*
	@Deprecated
	public CNN (float input[][], float output[][]) {
		int hiddens = Config.HIDDENS;
		this.input = input;
		this.output = output;

		layer = new Layer[Config.HIDDENS + 1];

		hidden = new float[hiddens][][];

		hidden[0] = new float[Config.HIDDEN_DEPTH_1][Config.HIDDEN_X_1*Config.HIDDEN_Y_1];

		Filter[] filter1 = Filter.getArrayFilter(Config.HIDDEN_DEPTH_1, Config.INPUT_FIELD, Config.INPUT_FIELD, Config.INPUT_DEPTH);

		layer[0] = new LayerConv(input, Config.INPUT_X, Config.INPUT_Y, Config.INPUT_DEPTH,
				hidden[0], Config.HIDDEN_X_1, Config.HIDDEN_Y_1, Config.HIDDEN_DEPTH_1,
				Config.INPUT_PADX, Config.INPUT_PADY,
				Config.INPUT_STRIDE, Config.INPUT_DILUTE,
				filter1, Config.RELU, null
				);


		for (int l = 1; l < hiddens; l ++) {
			hidden[l] = new float[Config.HIDDEN_DEPTH_1][Config.HIDDEN_X_1*Config.HIDDEN_Y_1];
			Filter[] filter2 = Filter.getArrayFilter(Config.HIDDEN_DEPTH_2, Config.HIDDEN_FIELD_1, Config.HIDDEN_FIELD_1, Config.HIDDEN_DEPTH_1);

			layer[l] =
					new LayerConv(hidden[l-1], Config.HIDDEN_X_1, Config.HIDDEN_Y_1, Config.HIDDEN_DEPTH_1,
							hidden[l], Config.HIDDEN_X_2, Config.HIDDEN_Y_2, Config.HIDDEN_DEPTH_2,
							Config.HIDDEN_PADX_1, Config.HIDDEN_PADY_1,
							Config.HIDDEN_STRIDE_1, Config.HIDDEN_DILUTE_1,
							filter2, Config.RELU, null
							);
		}

		Filter[] filtero = Filter.getArrayFilter(Config.OUTPUT_DEPTH, Config.HIDDEN_FIELD_2, Config.HIDDEN_FIELD_2, Config.OUTPUT_DEPTH);

		layer[hiddens] =
				new LayerConv(hidden[hiddens-1], Config.HIDDEN_X_2, Config.HIDDEN_Y_2, Config.HIDDEN_DEPTH_2,
						output, Config.OUTPUT_X, Config.OUTPUT_Y, Config.OUTPUT_DEPTH,
						Config.HIDDEN_PADX_2, Config.HIDDEN_PADY_2,
						Config.HIDDEN_STRIDE_2, Config.HIDDEN_DILUTE_2,
						filtero, Config.RELU, null
						);
	}
	*/
}

