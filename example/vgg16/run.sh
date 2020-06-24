# download vgg16 model and upgrade proto and caffemodel files
sh model/vgg16/net_file_upgrade.sh

# generate scale of weight and activation use quantation tools
sh example/vgg16/run_scale_quantation.sh

# get ncnn param bin from scale table
sh example/vgg16/run_caffe2ncnn.sh

# infer layer blob shape
sh example/vgg16/run_infer_shape.sh

# run scale fine tuning
sh example/vgg16/run_scale_fine_tuning.sh

# run validtion on imagenet val
sh example/vgg16/run_validation.sh
