# Generative Residual Convolutional Neural Network

代码主要来自 https://github.com/skumra/robotic-grasping ，即论文 [Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network](https://arxiv.org/abs/1909.04810). `src/inference` 来自原存储库中的 `inference`。

## 安装依赖

  conda create -n grcnn python=3.10 pytorch torchvision numpy scipy scikit-image rospkg

## 运行节点

    roslaunch grcnn grasp_planner.launch
    
## 运行测试

    roslaunch grcnn test.launch

`provider.py`从指定的话题截取RGB和深度图并调用`processor.py`提供的推理服务。
