import os, os.path

from optparse import OptionParser

parser = OptionParser()

parser.add_option("-i", "--index", dest="index", help="input weight file name")
(options, args) = parser.parse_args()
 
# get weight, log name
weight_name = "model_frcnn_" + options.index
os.system(
	"python test_frcnn.py" + 
	" -p new_data/2_is_sandal_data/shoesSamples/forTest/"
	 +
	 " --config_filename configs/" + weight_name + ".pickle" +
	)