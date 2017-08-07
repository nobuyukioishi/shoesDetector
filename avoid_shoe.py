# train + test
import os, os.path

# epoch_num = 400
# path="weights"
# num = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
# # get weight, log name
# weight_name = "model_frcnn_"+str(num)
# os.system(
# 	"python train_frcnn.py" + 
# 	" -p data/2_sss_data/"+
# 	" --num_epochs " + str(epoch_num) +
# 	" --hf 1" +
# 	" --vf 1" +
# 	" --output_weight_path weights/" + weight_name + ".hdf5" +
# 	" --config_filename configs/" + weight_name + ".pickle" +
# 	" >> logs/"+weight_name+".txt"
# 	)
num = 15
weight_name = "model_frcnn_"+str(num)
os.system("mkdir result_imgs/"+weight_name)
os.system(
	"python with_output_dir_test_frcnn.py" + 
	" -p data/2_sss_data/shoesSamples/forTest/"+
	" --config_filename configs/" + weight_name + ".pickle" +
	" --img_output result_imgs/"+ weight_name
	)