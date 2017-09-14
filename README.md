# Shoes Detector
F-RFCNN for detecting 3 kind of objects(Sandal, shoes, slipper).
## Folders
We saved training, test images and annotations in *data/*. 
## Training and test
We used *combo_master.py* to controls the slave script *train_frcnn.py* and *with_output_dir_test_frcnn.py*. As a result we can train multiple training and test phase in one run.
## Result
combo_master.py's result for detecting test pictures are saved in *result_imgs/*. Also training log, best score weight are saved in *logs/* and *weights/*. Saving name *model_frcnn_X* have an index *X* which is the identity for log, weight, pickle file. 

Example in file naming: 

weight: model_frcnn_5.hdf5  

log: model_frcnn_5.txt 

result image directory: model_frcnn_5/
