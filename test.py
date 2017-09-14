import os
import numpy as np
import cv2

img = cv2.imread("data/lungCancer_frcnn/data/1_lung_cancer_only_cancer_labeled/shoesSamples/PNGImages/1_IMG28.jpg")



# for validating result_imgs
# import os
# # Output predicts of input directory.
# # input: output directory, weight index
# path="weights"
# num = 11  # index number of weight file
# weight_name = "model_frcnn_"+str(num)

# os.system("mkdir result_imgs/"+weight_name)
# os.system(
# 	"python with_output_dir_test_frcnn.py" + 
# 	" -p data/2_sss_data/shoesSamples/forTest/for_confusion_matrix/2017_7_5"+
# 	" --output_config_filename configs/" + weight_name + ".pickle" +
# 	" --img_output result_imgs/"+ weight_name
# 	)

# # for validating result_imgs_old 
# import os
# # Output predicts of input directory.
# # input: output directory, weight index
# path="weights"
# num = 11  # index number of weight file
# weight_name = "model_frcnn_"+str(num)

# os.system("mkdir result_imgs_old/"+weight_name)
# os.system(
# 	"python with_output_dir_test_frcnn.py" + 
# 	" -p data/2_sss_data/shoesSamples/forTest"+
# 	" --output_config_filename configs/" + weight_name + ".pickle" +
# 	" --img_output result_imgs_old/"+ weight_name
# 	)


# for open_campus output
# import os
# # Output predicts of input directory.
# # input: output directory, weight index
# path="weights"
# num = 11  # index number of weight file
# weight_name = "model_frcnn_"+str(num)

# os.system("mkdir open_campus/"+weight_name)
# os.system(
# 	"python with_output_dir_test_frcnn.py" + 
# 	" -p data/2_sss_data/shoesSamples/forTest/open_campus"+
# 	" --output_config_filename configs/" + weight_name + ".pickle" +
# 	" --img_output open_campus/"+ weight_name
# 	)
