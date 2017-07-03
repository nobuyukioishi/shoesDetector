import os, os.path

# number of file in weight directory
num = len([name for name in os.listdir('./weights') if os.path.isfile(name)])
# get weight, log name
weight_name = "model_frcnn_"+str(num)
os.system(
	"python train_frcnn.py" + 
	" -p new_data/2_is_sandal_data/"+
	" --num_epochs 1000" +
	" --output_weight_path weights/" + weight_name + ".hdf5" +
	" --config_filename configs/" + weight_name + ".pickle" +
	" >> logs/"+weight_name+".txt"
	)
