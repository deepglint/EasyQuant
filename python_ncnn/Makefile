BOOST_INCLUDE_PATH=/usr/include
BOOST_LIB_PATH=/usr/lib/x86_64-linux-gnu
NCNN_INCLUDE_PATH=/train/trainset/1/docker_run/ncnn/build/install/include/ncnn
NCNN_LIB_PATH=/train/trainset/1/docker_run/ncnn/build/install/lib
PYTHON_LIB_PATH=/usr/lib/x86_64-linux-gnu
PYTHON_INCLUDE_PATH=/usr/include/python2.7
NUMPY_INCLUDE_PATH=/usr/local/lib/python2.7/dist-packages/numpy/core/include

all:
	g++ -fPIC -fpermissive -I ${BOOST_INCLUDE_PATH} -I ${PYTHON_INCLUDE_PATH} -I ${NUMPY_INCLUDE_PATH} -I ${NCNN_INCLUDE_PATH} -c ncnn.cpp 
	g++ -fPIC -fpermissive -I ${BOOST_INCLUDE_PATH} -I ${PYTHON_INCLUDE_PATH} -I ${NUMPY_INCLUDE_PATH} -I ${NCNN_INCLUDE_PATH} -c ncnn2python.cpp 
	g++ -shared  -o ncnn.so  ncnn.o ncnn2python.o -L ${BOOST_LIB_PATH} -lboost_python -L ${PYTHON_LIB_PATH} -lpython2.7  -L ${NCNN_LIB_PATH} -lncnn -lgomp
	cp ncnn.so /usr/local/lib/python2.7/dist-packages/
	

clean:
	rm ncnn.so  ncnn.o ncnn2python.o
