# 安全与隐私保护联邦学习系统

随着人工智能的发展，集中式学习系统存在隐私泄露以及数据孤岛等难以解决的
难题。在此背景下，能够让参与者保留训练数据的同时共享学习结果的联邦学习愈发受到重视。然而联邦学习系统仍然面临着其本身及来自攻击者的安全和隐私泄漏隐患，因此有必要面向联邦学习系统保护其安全性与隐私性。
本项目基于以上背景， 基于Python开发了一个安全与隐私保护的联邦学习系统。
该系统针对多种主流的联邦学习攻击，实现相应的防御方案并确保数据的隐私性和模型的鲁棒性。 该系统中的防御方案采取模块化设计，可以独立为一个安全与隐私保护工具库并应用于其他联邦学习系统。同时，该系统可以根据不同安全需求灵活选择并添加防御模块。
系统分为四个安全模块： 

* 拜占庭容错：拜占庭容错模块对节点上传梯度进行比较处理来对拜占庭攻击进行防御
* 后门攻击防御：后门攻击防御模块首先对节点提交模型进行缩放，再向模型添加噪声来防御后门攻击
* 数据恢复攻击防御： 数据恢复攻击防御模块可以通过对训练图像施加变换策略对数据恢复攻击进行防御
* 差分隐私：差分隐私模块在迭代训练时过梯度添加噪声等操作来保护数据隐私

## 安装环境

- torch (1.8.0)
- torchvision (0.9.0)
- numpy (1.19+)
- mpi4py (3.0.3+)
- argparse (1.4.0+)
- scipy (1.6.3+)
- logging (0.4.9.6+)

## 安装教程

```bash
git clone git@github.com:csl-cqu/Federated-Learning-System.git
pip install -r requirements.txt
```

## 架构

![Architecture](assets/Architecture.png)

每个包具体大体功能如下：

- **assets**: 图片存放
- **data**: CIFAR10, CIFAR100, MNIST数据集获取
- **modules**: backdoor_defense, byzantine_resilience, data_inference_defense, differential_privacy 模块实现
- **core**: 分布式计算实现
- **api**: 通过调用core实现联邦学习算法
- **experiment**: GPU映射配置
- **sh**: 不同模块测试脚本

## 使用说明

使用MNIST数据集上使用Logistic回归模型测试来演示

1. 下载数据集

   ```bash
   bash data/mnist/download_and_unzip.sh
   ```

2. 选择/添加GPU映射，查看

   [spfl_experiments/distributed/fedavg/GPU_MAPPING.md](spfl_experiments/distributed/fedavg/GPU_MAPPING.md)  

3. 运行测试脚本

   ```bash
   bash test/baseline_mnist_lr.sh
   ```

### 部分参数说明

- **data_dir**: 数据集位置
- **partition_method**: 数据集分割方式
- **client_num_in_total**: 节点总数
- **client_num_per_round**: 每轮选取的节点数量
- **client_optimizer**: 节点优化器
- **backend:** 通信后端
- **wandb-off:** 禁用wandb

`test` 文件夹中包含了其它数据集测试以及四种防御模块的单独脚本，如果想对参数进行改动请参考每个模块说明文件：

| 参数                         | 模块                      | README                                                       |
| ---------------------------- | ------------------------- | ------------------------------------------------------------ |
| `--byzantine_aggregate`      | Byzantine Fault Tolerance | [modules/byzantine_resilience/README.md](modules/byzantine_resilience/README.md) |
| `--backdoor-defense`         | Backdoor Defense          | [modules/backdoor_defense/README.md](modules/backdoor_defense/README.md) |
| `--data_recovery_protection` | Data Recovery Defense     | [modules/data_inference_defense/README.md](modules/data_inference_defense/README.md) |
| `--differential_privacy`     | Differential Privacy      | [modules/differential_privacy/README.md](modules/differential_privacy/README.md) |

## 致谢

### 项目贡献者
<a href="https://github.com/barryZZJ"><img src="/assets/barryzzj.png" width="50px"></a> <a href="https://github.com/endereye"><img src="/assets/endereye.png" width="50px"></a> <a href="https://github.com/Du11JK"><img src="/assets/du11jk.png" width="50px"></a> <a href="https://github.com/LuckMonkeys"><img src="/assets/luckmonkeys.png" width="50px"></a>

### 参考

> FedML: 
> https://github.com/FedML-AI/FedML
>
> NDSS21-Model-Poisoning
> https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning

更多信息，请运行：
```bash
python spfl_experiments/distributed/fedavg/main_fedavg.py -h
```
