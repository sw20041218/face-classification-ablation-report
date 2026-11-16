# 运行与复现

## 文件与数据

- `common/face_model.py`：原始对比度、HOG、单位化和MATLAB向量方向实现。
- `course800/`：800训练样本的E0—E10拟合、420无标签样本的提交预测、冻结校准常数、作者确认的平台ACC。
- `full31022/`：完整MIT CBCL数据准备、CNN特征提取、SVM拟合与内部校准、冻结评价、结果及验证记录。
- `docs/人脸分类实验报告.docx`：与根目录README对应的公开报告。

不上传私有人工参考标签、原始课程数据、CNN特征缓存、模型、虚拟环境或历史无关搜索代码。完整数据集从固定公开源码提交下载，遵守上游数据使用说明。

## 环境

历史SVM环境：Python 3.12、NumPy 1.26.4、SciPy 1.13.1、scikit-learn 1.5.2、joblib 1.4.2。
历史完整集CNN环境：Python 3.10、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、NumPy 2.2.6。见`full31022/*environment.json`。
建议分别建立`svm`和`features`环境，分别安装`requirements.txt`和`requirements-features.txt`；CUDA版本应与机器驱动匹配。下列`python`表示已激活该步骤对应的环境。完整集特征脚本要求CUDA；课程特征脚本使用CPU。

```bash
python -m pip install -r requirements.txt
# 在独立特征环境中
python -m pip install -r requirements-features.txt
python tools/download_assets.py --weights
```

## 800训练样本与420课程测试样本

自行准备下列文件，不需要也不读取`test_label.mat`：

```text
COURSE_DATA/
  train(Task_2)/train_data.mat   # train_data: (800,361)
  train(Task_2)/train_label.mat  # train_label: 800 labels in {-1,+1}
  test(Task_2)/test_data.mat     # test_data: (420,361)
```

沿用原始MAT文件像素尺度及MATLAB按列展开约定，不要在外部另改归一化或图像方向。

```bash
# 特征环境，从仓库根目录执行
python course800/extract_features.py --data "COURSE_DATA"
# SVM环境
python course800/run_ablation.py --data "COURSE_DATA" --out "course800/outputs"
```

生成E0—E10的`*_predictions.txt`，每行`编号 标签`，编号从1开始。提交平台后才能取得新的平台ACC。`platform_acc.csv`是作者确认的已有平台成绩，脚本不自动获得平台反馈。

`frozen_calibration.json`来自历史开发记录，逐组保留当时的分数标准差和阈值。历史配置及阈值曾使用人工参考标签选择，因而不能宣称整个方法开发从未接触测试来源信息。本公开入口仅重放冻结常数，不上传参考标签，不进行420样本的本地测试评价，也不重新搜索阈值。重新提取特征时，硬件或依赖差异可能影响临界样本；不要据此自动改阈值。

## 完整31022张MIT CBCL

```bash
python tools/download_assets.py --dataset
# SVM环境
python full31022/prepare_dataset.py
# CUDA特征环境；生成缓存需要数GB磁盘空间
python full31022/extract_features.py --batch 16
# SVM环境
python full31022/run_full_ablation.py --stage fit
python full31022/run_full_ablation.py --stage evaluate
python full31022/verify_full.py
```

官方训练6977张按精确像素与水平镜像分组，以固定种子划分5580拟合及1397内部验证校准。官方测试24045张仅在配置冻结后用于本轮评价。保留原始官方划分，历史同源420样本影响过方法开发，不宣称从未接触的盲测独立性。

`--stage fit`仅用内部验证标签校准阈值及标准差，不在校准后重拟合；`--stage evaluate`读取冻结配置进行测试。测试非人脸占98.04%，必须结合召回率、F1、AUC及FP/FN解读ACC。

仓库发布既有结果、预测、融合分数及冻结配置，未发布大体积模型/CNN缓存。重跑前请把仓库复制到独立工作目录；缺失模型时会重新拟合，避免把不同运行产生的缓存和结果混用。完整SVM核矩阵约1GB，实际内存需求更高。

## 来源

数据镜像：[galeone/mlcnn 的固定提交](https://github.com/galeone/mlcnn/tree/a6bfe4e6e8a75a9ca1f4e216ef96be7253efdbb3/mitcbcl)。下载脚本验证ZIP提交标识；`download_manifest.json`记录历史下载压缩包的哈希，固定提交ZIP的封装哈希可能不同，图像清单由数据准备脚本重建。

预训练权重使用torchvision官方ImageNet V1文件，下载后校验文件名中的SHA256前缀。
