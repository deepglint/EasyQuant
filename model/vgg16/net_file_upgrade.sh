# download vgg16 model files from model pool
wget -P model/vgg16/ https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt
wget -P model/vgg16/ http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

# upgrade proto and bin model file
caffe/build/tools/upgrade_net_proto_text model/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt model/vgg16/VGG_ILSVRC_16_layers_deploy_new.prototxt
caffe/build/tools/upgrade_net_proto_binary model/vgg16/VGG_ILSVRC_16_layers.caffemodel model/vgg16/VGG_ILSVRC_16_layers_new.caffemodel
