# train + test
import os, os.path

"""
This script automatically generates training, testing shell command which will search 
over epoch and rpn_min, rpn_max overlap parameters.

First loop will grid search by epoch.

Second loop will search by rpn_min_overlap, self.rpn_max_overlap.

Ps: there is high dependency on test dataset, train dataset, config path. First you should check it.
"""


class PhaseScriptGenerator(object):
    """
    This script automatically generates training, testing shell command which will search
    over epoch and rpn_min, rpn_max overlap parameters.

    Second loop will search by rpn_min_overlap, self.rpn_max_overlap.

    """
    
    def __init__(self, dataset_path, testset_path, weight_path="weights"):
        self.data_set_path = dataset_path
        self.test_set_path = testset_path
        self.weight_path = weight_path  # "weights"

    def grid_search_epoch(self, first_epoch, last_epoch, range_epoch):
        print("training: range(", first_epoch, last_epoch, range_epoch, ")")
        for epoch_num in range(first_epoch, last_epoch, range_epoch):
            path = self.weight_path
            num = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
            # get weight, log name
            weight_name = "model_frcnn_" + str(num)
            os.system(
                "python train_frcnn.py" +
                " -p " + self.data_set_path +
                " --num_epochs " + str(epoch_num) +
                " --hf 1" +
                " --vf 1" +
                " --output_weight_path weights/" + weight_name + ".hdf5" +
                " --output_config_filename configs/" + weight_name + ".pickle" +
                # " --load_config_filename " + load_config_filename +
                " >> logs/" + weight_name + ".txt"
            )
            os.system("mkdir imgs/" + weight_name)
            os.system(
                "python with_output_dir_test_frcnn.py" +
                " -p " + self.test_set_path +
                " --output_config_filename configs/" + weight_name + ".pickle" +
                " --img_output imgs/" + weight_name
            )

    def epoch_test_only(self, first_weight_id, last_weight_id):

        for weight_id in range(first_weight_id, last_weight_id):
            weight_name =  weight_name = "model_frcnn_" + str(weight_id)
            os.system("mkdir imgs/" + weight_name)
            os.system(
                "python with_output_dir_test_frcnn.py" +
                " -p " + self.test_set_path +
                " --output_config_filename configs/" + weight_name + ".pickle" +
                " --img_output imgs/" + weight_name
            )

    def grid_search_rpn(self, number_epoch):
        for config_num in range(1,8):
            path = self.weight_path
            num = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
            # get weight, log name
            weight_name = "model_frcnn_"+str(num)
            os.system(
                "python train_frcnn" + str(config_num) + ".py" +
                " -p " + self.dataset_path +
                " --num_epochs " + str(number_epoch) +
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
                " -p " + self.testset_path +
                " --output_config_filename configs/" + weight_name + ".pickle" +
                " --img_output imgs/"+ weight_name
                )
            
            print("Test imgs: ", self.test_set_path)
            print("Output path: ", "imgs/", weight_name)


       

data_set_path = "data/3_only_slipper_in_sss/"
test_set_path = "data/3_only_slipper_in_sss/shoesSamples/forTest/"
weight_path = "weights"


generator = PhaseScriptGenerator(dataset_path=data_set_path,  testset_path=test_set_path, weight_path=weight_path)
# generator.epoch_test_only(1,6)
generator.grid_search_epoch(400, 401, 100)
# generator.grid_search_rpn(number_epoch=300)
