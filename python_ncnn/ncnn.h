#include "net.h"
#include "cpu.h"
#include "benchmark.h"
#include <string>
#include <boost/python.hpp> 
#include <numpy/arrayobject.h> 

using namespace boost::python; 
namespace bp = boost::python; 
using namespace std;


class Net{
public:
	Net();
	~Net();
	int load_param(const char * paramPath);
	int load_model(const char * modelPath);
	void setInputBlobName(string name);
	void setOutputBlobName(string name);
	int inference(object & input_object,object & output_object,int inputHeight,int inputWidth);
	int inference_debug_writeOutputBlob2File(object & input_object,object & output_object,int inputHeight,int inputWidth);
private:
	ncnn::Net net;
	string inputBlobNmae;
	string outputBlobName;

};