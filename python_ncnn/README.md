1. 安装依赖包
	
```shell
	sudo apt-get install libboost-python-dev
	```
	
2. 修改 Makefile 开头的变量路径；


3. 执行 make 命令编译 ncnn.so；
4. 把 ncnn.so 放到 site-packages 目录下( /home/zyy/anaconda3/lib/python3.7/site-packages )，在 python 环境下 import ncnn 进行测试，不报错即加载成功。

