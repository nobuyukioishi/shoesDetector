import os, os.path
path="weights"
num = 11

weight_name = "model_frcnn_"+str(num)
os.system(
    "python viz_frcnn.py" + 
    " -p data/2_sss_data/shoesSamples/forTest/"+
    " --output_config_filename configs/" + weight_name + ".pickle"
    )
