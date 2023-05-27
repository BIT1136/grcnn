# Generative Residual Convolutional Neural Network

https://github.com/skumra/robotic-grasping 的 ROS 包装，代码也主要来自此仓库；原论文为 [Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network](https://arxiv.org/abs/1909.04810). `src/inference` 来自原存储库中的 `inference`。

## 安装依赖

    conda create -c conda-forge -n grcnn python=3.10 pytorch=1.13 torchvision=0.14 numpy=1.24 scipy=1.10 scikit-image=0.20 rospkg=1.5

## 运行节点

    roslaunch grcnn grasp_planner.launch
    
## 运行测试

    roslaunch grcnn test.launch

按回车时`provider.py`从指定的话题截取RGB和深度图并调用`processor.py`提供的推理服务。

## 训练模型

使用原仓库并遵循其说明以进行训练，依赖安装方式为：

    conda create -c conda-forge -n grcnn_train python numpy=1.19 pytorch torchvision opencv scikit-image tensorboardx matplotlib pillow
    pip install torchsummary

使用numpy=1.19以兼容原仓库代码中的`np.int`
