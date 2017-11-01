# train + test
import os, os.path

"""
This script automatically generates training, testing shell command which will search 
over epoch and rpn_min, rpn_max overlap parameters.

First loop will grid search by epoch.

Second loop will search by rpn_min_overlap, self.rpn_max_overlap.

Ps: there is high dependency on test dataset, train dataset, config path. First you should check it.
"""


dataset_path = "data/3_only_slipper_in_sss/"
testset_path = "data/3_only_slipper_in_sss/shoesSamples/forTest/"

# for epoch_num in range(1,2):  # for debug
for epoch_num in range(100,700,100):
	# number of file in weight directoryf
	path="weights"
	num = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
	# get weight, log name
	weight_name = "model_frcnn_"+str(num)
	os.system(
		"python train_frcnn.py" + 
		" -p " + dataset_path +
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
		" -p "+ testset_path +
		" --output_config_filename configs/" + weight_name + ".pickle" +
		" --img_output imgs/"+ weight_name
		)

# RPN's training 
# todo : add to sheet
for config_num in range(1,8):
	path="weights"
	num = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
	# get weight, log name
	weight_name = "model_frcnn_"+str(num)
	os.system(
		"python train_frcnn" + str(config_num) + ".py" + 
		" -p " + dataset_path +
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
		" -p " + testset_path +
		" --output_config_filename configs/" + weight_name + ".pickle" +
		" --img_output imgs/"+ weight_name
		)
