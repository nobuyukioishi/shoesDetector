# train + test
import os, os.path

# for epoch_num in range(1,2):  # for debug
for epoch_num in range(100,700,100):
	# number of file in weight directoryf
	path="weights"
	num = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
	# get weight, log name
	weight_name = "model_frcnn_"+str(num)
	os.system(
		"python train_frcnn.py" + 
		" -p data/2_sss_data/"+
		" --num_epochs " + str(epoch_num) +
		" --hf 1" +
		" --vf 1" +
		" --output_weight_path weights/" + weight_name + ".hdf5" +
		" --output_config_filename configs/" + weight_name + ".pickle" +
		# " --load_config_filename " + load_config_filename +
		" >> logs/"+weight_name+".txt"
		)
	os.system("mkdir imgs/"+weight_name)
	os.system(
		"python with_output_dir_test_frcnn.py" + 
		" -p data/2_sss_data/shoesSamples/forTest/"+
		" --config_filename configs/" + weight_name + ".pickle" +
		" --img_output imgs/"+ weight_name
		)

# RPN's training 
# todo : add to sheet
for config_num in range(1,8):
	# number of file in weight directory
	path="weights"
	num = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
	# get weight, log name
	weight_name = "model_frcnn_"+str(num)
	os.system(
		"python train_frcnn" + str(config_num) + ".py" + 
		" -p data/2_sss_data/"+
		" --num_epochs 300" +
		" --hf 1" +
		" --vf 1" +
		" --output_weight_path weights/" + weight_name + ".hdf5" +
		" --output_config_filename configs/" + weight_name + ".pickle" +
		# " --load_config_filename " + load_config_filename +
		" >> logs/"+weight_name+".txt"
		)
	os.system("mkdir imgs/"+weight_name)
	os.system(
		"python with_output_dir_test_frcnn.py" + 
		" -p data/2_sss_data/shoesSamples/forTest/"+
		" --config_filename configs/" + weight_name + ".pickle" +
		" --img_output imgs/"+ weight_name
		)

