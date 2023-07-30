## 概述
小样本食物识别
## 时间线
### 06.08
- create this project
- complete the construction of the dataset class
### 07.08-07.11
- complete the project according to the framework of mmfewshot
### 07.12-07.13
- make the training process run succesfully
### 7.14
- complete the function that when a test interval is reached, we test the model
### 7.17-7.18
- borrow the idea from the anp
### 7.19-7.23
- try other idea, such as cnp, frn and cross-attention, but failed.
### 7.27-7.30
- try adaptive neural process successfully
## 遇到的问题
### 使用mmfewshot框架时遇到的问题
如果出现“THC/THC.h: No such file or directory”错误，需要调低pytorch版本到1.10，使用命令：
```
pip install torch==1.10.0 torchvision
```
最重要的是，根据mmfewshot的[官方文档](https://mmfewshot.readthedocs.io/en/latest/install.html)来
安装对应的其他框架，如：mmcls，mmdet，不能直接使用pip install mmcls进行安装，要指定版本。
经过一次一次的试错，安装mmfewshot环境的正确步骤为： 

(1)使用上面提到的命令安装pytorch 1.10，不能超过这个
(2)安装mmfewshot的环境
```
!pip install openmim
mim install mmcv-full==1.3.12
mim install mmcls==0.15.0
mim install mmdet==2.16.0
```
使用这些命令就能正确安装mmfewshot所需的环境了，之后就是根据自己的需要修改配置文件。

### 关于pip命令的领悟

### 分布式训练的问题
1.Error initializing torch.distributed using env:// rendezvous: environment variable RANK expected, but not set
在命令行运行即可
2.AttributeError: 'DistributedDataParallel' object has no attribute 'forward_train'
在网络前面增加module
3.Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
将参数输入到网络之前加上cuda()，注意加入口处就行，不然会报其他错误。
4.关掉ssh连接窗口出现如下错误：WARNING:torch.distributed.elastic.agent.server.api:Received 1 death signal, shutting down workers
使用tmux
```
tmux new -s train_model
训练命令
按下ctrl + b后，再按d键退出窗口
```
tmux attach -t train_model
### 训练问题
1.损失不下降
因为我冻结了模型，导致不训练。
### anp
1.RuntimeError: CUDA error: device-side assert triggered
生成one-hot编码出错，因为query label的值为：1 20 30 4 0 50这种，而class_num只设置为5，
需要将query label中的标签重新转换，范围变成0-class_num。
2.afn中的cross-attention中query和support的batch大小不一致，导致如果support和query大小不一致，
将无法运行
3.softmax得到的值很平均，损失不下降
4.cnp,anp试验失败，损失下降了但是过拟合了

## 结果记录
### 7.14
1.setting: use pretrain, optimzer:SGD lr:0.002 iterations:10K episode_num:2000 epoch:60 img_size:84 no_scheduler,
results: top-1:45.98%
### 7.15
1.setting: use pretrain, optimzer:SGD lr:0.002 iterations:20K episode_num:2000 epoch:100 img_size:224 no_scheduler,
5-way 1-shot results: top-1:48.50% 5-way 5-shot results: top-1:64.83%
2.setting optimzer: use pretrain,SGD lr:0.002 iterations:20K episode_num:2000 epoch:100 img_size:224 40,80 decrease lr by 0.1,
5-way 1-shot results: top-1:46.40%
### 7.16
1.setting: optimzer: no pretrain, SGD lr:0.002 iterations:20K episode_num:2000 epoch:100 img_size:224 no_scheduler,
5-way 1-shot results: top-1:48.63% 5-way 5-shot results: 
### 7.17
setting: optimzer: no pretrain, AdawW lr:0.002 iterations:20K episode_num:2000 epoch:100 img_size:224 no_scheduler,
5-way 1-shot results: top-1:52.58% 5-way 5-shot results: 70.51%
### 7.25
method :canp setting: optimzer: pretrain, SGD lr:0.002 iterations:20K episode_num:2000 epoch:100 img_size:224 60 decrease lr by 0.1,
5-way 1-shot results: top-1:29.33%

### 7.27
setting: optimzer: pretrain, SGD lr:0.0001 iterations:20K episode_num:2000 epoch:100 img_size:224 no decrease,
5-way 1-shot results: top-1(epoch=60):34.91% top-1(epoch=100):38.20%
### 7.27
setting: optimzer: pretrain, AdaW lr:0.0001 iterations:40K episode_num:2000 epoch:200 img_size:224 150 decrease lr by 0.1,
5-way 1-shot results: top-1(epoch=185):56.84% top-1(epoch=100 no decrease): 53.73%
### 7.28
setting: optimzer: pretrain, AdaW lr:0.0001 iterations:20K episode_num:2000 epoch:100 img_size:224 75 decrease lr by 0.1,
5-way 1-shot results: top-1(epoch=100):55.19%

### 7.29
setting: optimzer: pretrain, AdaW lr:0.0001 iterations:20K episode_num:2000 epoch:100 img_size:224 60 decrease lr by 0.1,
5-way 1-shot results: top-1(epoch=100):54.03%

### 7.29
setting: optimzer: pretrain, AdaW lr:0.0004 iterations:20K episode_num:2000 epoch:100 img_size:224 75 decrease lr by 0.1,
5-way 1-shot results: top-1(epoch=100):50.16%
### 7.30
setting: optimzer: pretrain, AdaW lr:0.0001 iterations:20K episode_num:2000 epoch:100 img_size:224 75 decrease lr by 0.1 wd:0.01
5-way 1-shot results: top-1(epoch=100):53.89%

## 经验记录
### few shot classification数据集构建
1.文件结构应该为：
```
- data_name
--- images
----- folder_name1
------- img1.png
------- img2.png
----- folder_name2
--- meta
----- classes.txt  
----- fsl_train.txt
----- fsl_test.txt
----- fsl_train_class.txt
----- fsl_test_class.txt
```
其中folder_name1,folder_name2是文件夹的名字，通常是分类名称，有些可能也是下标数字(1-100的数字)
2.找到图像的标签文件classes.txt，里面含有图像全部的类别,如果没有需要自己构建一个
3.使用下面的文件生成fsl_train.txt,fsl_test.txt,fsl_train_class.txt,fsl_test_class.txt文件
该代码目前支持的情况有：
- (1) folder_name为类别名称
- (2) folder_name为类别名称对应的下标，从1开始
- (3) folder_name文件夹下面的图片名称全部是数字，没有其他符号
- (4) folder_name文件夹下面的图片名称什么符号都有

4.构建basedataset类，用于加载含有类别名称的文件

5.构建通用的少样本数据集加载类,UniversalFewShotDataset

6.构建针对元学习的数据集加载类,UniversalEpisodeDataset

7.构建自己的配置文件，我使用的是json格式的
