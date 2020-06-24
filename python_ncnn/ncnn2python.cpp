#include "ncnn.h"

BOOST_PYTHON_MODULE(ncnn) 
{ 
 	class_<Net>("net",init<>())
 	.def("load_param",&Net::load_param)
 	.def("load_model",&Net::load_model)
 	.def("setInputBlobName",&Net::setInputBlobName)
 	.def("setOutputBlobName",&Net::setOutputBlobName)
 	.def("inference",&Net::inference)
	.def("inference_debug_writeOutputBlob2File",&Net::inference_debug_writeOutputBlob2File)
 	;
}