# Generative Residual Convolutional Neural Network

代码主要来自https://github.com/skumra/robotic-grasping，即论文[Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network](https://arxiv.org/abs/1909.04810)

运行`install_env.sh`创建conda环境并安装依赖，再将scripts中脚本的Shebang修改为环境中的解释器路径。

`provider.py`从奥比中光相机截取RGB和深度图并调用`processor.py`提供的推理服务。
