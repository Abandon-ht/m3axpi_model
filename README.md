# [m3axpi] YOLOv5训练到部署全流程
### 介绍
YOLOv5 是<a href="https://ultralytics.com"> Ultralytics </a>公司于 2020 年 6 月 9 日开源的 2d 实时目标检测算法。涵盖 yolov5n、yolov5n6、yolov5s、yolov5s6、yolov5m、yolov5m6、yolov5l、yolov5l6、yolov5x、yolov5x6 等十几个模型。YOLOv5 具有训练速度快、推理时间短、非常易于训练、方便部署等优点。YOLOv5 的网络结构可以分为intput、backbone、neck、head 四个部分。

本教程讲解 YOLOv5 的训练并用 Sipeed 公司的开发板 m3axpi 部署。

### 开始
首先创建工作目录,以下所有操作均在此目录内。

右键打开终端,执行以下命令:
```bash
cd ~
mkdir m3axpi && cd m3axpi
```

### 一、准备数据集
如何制作目标检测数据集请参考(链接),本教程用标注好的“垃圾检测“数据集对整个流程进行讲解。该数据集可以通过以下三种方式获取:

1.直接下载
百度网盘:
Github: https://github.com/Abandon-ht/coco_rubbish_dataset/archive/refs/heads/main.zip
将下载的数据集解压到 datasets 文件夹,并重命名为 rubbish

2.拉取数据集的仓库
```bash
mkdir datasets && cd datasets
git clone https://github.com/Abandon-ht/coco_rubbish_dataset.git rubbish
```
3.终端命令下载
```bash
mkdir datasets && cd datasets
wget https://github.com/Abandon-ht/coco_rubbish_dataset/archive/refs/heads/main.zip
unzip coco_rubbish_dataset-main.zip
mv coco_rubbish_dataset-main rubbish
```

三种方法都可以得到以下2个文件夹和3个文件

![](./images/002.png)


### 二、获取 yolov5 源码
在 m3axpi 目录下(注意不是在 datasets 目录下),拉取 yolov5 源码的仓库

```bash
cd ~/m3axpi
git clone -b v6.2 https://github.com/ultralytics/yolov5.git  # clone
cd yolov5
pip install -r requirements.txt  # install
```
![](./images/003.png)

yolov5 目录如图所示:

![](./images/004.png)


### 三、训练 yolov5 模型

进入 yolov5 的工作目录,
复制一份 coco.yaml,并重命名为 rubbish.yaml

```bash
cp data/coco.yaml data/rubbish.yaml
```
根据图片修改垃圾分类数据集的路径和类别名字
```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/rubbish  # dataset root dir
train: train2017.txt  # train images (relative to 'path') 118287 images
val: val2017.txt  # val images (relative to 'path') 5000 images
# test: test-dev2017.txt  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# Classes
names:
  0: battery
  1: pills
  2: bananas
  3: apples
  4: cans
  5: bottles
  6: ceram
  7: green vegatable
  8: broccoli
  9: boxes
  10: glass
  11: cigarette
  12: orange
  13: eggshell
  14: spitball
  15: tile
```

![](./images/005.png)

修改完成以后,用以下命令开始训练 yolov5s 模型

```bash
python train.py --data data/rubbish.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --batch-size -1 --epoch 20
```
![](./images/006.png)

数据集加载成功,开始训练模型。如果没有加载成功,请检查数据集路径

![](./images/007.png)

训练完成后,可以在 ./runs/train/exp/ 文件夹下查看训练的日志


PR_curve.png 为 mAP_0.5 曲线
![](./images/PR_curve.png)

results.png 为全部曲线
![](./images/results.png)

### 四、模型预测和导出
可以使用以下命令预测图片,注意需要将图片和模型的路径修改为自己的路径
```bash
python detect.py --source ../datasets/rubbish/images/IMG_20210311_213716.jpg --weights ./runs/train/exp/weights/best.pt
```
![](./images/008.png)

可以在 runs/detect/exp 目录下。看到预测的图片

![](./images/009.jpg)


使用以下命令导出 onnx 模型,注意加上 opset=11 这个参数

![](./images/010.png)

```bash
python export.py --include onnx --opset 11 --weights./runs/train/exp/weights/best.pt
```
导出的 onnx 模型在 runs/train/exp/weights 目录下

![](./images/010a.png)

在浏览器地址栏中输入 netron.app (这个地址),把上一步导出的 best.onnx 文件拖进去,查看 yolov5 的模型结构。

模型的输入名字为:images

![](./images/011a.png)

使用 torch1.13.0 之前版本导出的模型,最后的三个卷积(Conv)输出是以 onnx::Reshape_329 结尾。三个卷积(Conv)输出的结尾数字不相同。你导出的模型结尾的数字可能与我的不相同。


第一个 Conv 的输出是 onnx::Reshape_329

![](./images/012a.png)

第二个 Conv 的输出是 onnx::Reshape_367

![](./images/013a.png)

第三个 Conv 的输出是 onnx::Reshape_405

![](./images/014a.png)

### 五、修改 ONNX 模型
1.脚本修改 ONNX 模型
可以使用以下脚本修改模型,注意如果用 torch1.13.0 及之后版本导出的模型,请使用下面介绍方法2图形化修改模型

```python
import argparse
import onnx

def onnx_sub():
    onnx.utils.extract_model(opt.onnx_input, opt.onnx_output, opt.model_input, opt.model_output)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_input', type=str, default='weights/yolov5s.onnx', help='model.onnx path(s)')
    parser.add_argument('--onnx_output', type=str, default='weights/yolov5s_sub.onnx', help='model_sub.onnx path(s)')
    parser.add_argument('--model_input', '--input', nargs='+', type=str, default=["images"], help='input_names')
    parser.add_argument(
        '--model_output', '--output', nargs='+', type=str, default=["onnx::Reshape_329",
                                                                    "onnx::Reshape_367",
                                                                    "onnx::Reshape_405"], help='output_names')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    sub = onnx_sub()

```
新建一个名字为 onnxcut.py 的文件,复制以上代码

使用以下命令,注意一定要将模型的三个输出,替换为自己模型最后三个卷积(Conv)的输出

```bash
python onnxcut.py --onnx_input ./runs/train/exp/weights/best.onnx --onnx_output ./runs/train/exp/weights/best_cut.onnx --model_input images --model_output onnx::Reshape_329 onnx::Reshape_367 onnx::Reshape_405                                                                                                
```

![](./images/015.png)

修改后的模型如同所示,三个输出分别为 onnx::Reshape_329、onnx::Reshape_367、onnx::Reshape_405

![](./images/016a.png)

2.图形化修改 ONNX 模型

也可以使用图形化修改模型,使用以下命令拉取源码仓库

```bash
cd ~/m3axpi
git clone https://github.com/ZhangGe6/onnx-modifier.git
```

![](./images/016b.png)

目录如图所示:

![](./images/017.png)

进入目录,使用以下命令安装依赖包

```bash
cd onnx-modifier
pip install onnx flask
```
![](./images/017a.png)

使用以下命令运行脚本
```bash
python app.py
```
![](./images/017b.png)

在浏览器地址栏输入
http://127.0.0.1:5000
打开网页,把刚才第三步导出的 best.onnx 文件拖进去, 修改模型结构。

依次选中最后三个卷积(Conv)之后的Reshape, 点击 Delete With Children


![](./images/018a.png)

选中第一个删除 Reshape 上方的 Conv,修改 OUTPUTS 的名字为 Conv_output_0,点击 Add Output 添加输出


![](./images/019a.png)

选中第二个 Reshape,点击 Delete With Children


![](./images/020a.png)

选中第二个删除 Reshape 上方的 Conv,修改 OUTPUTS 的名字为 Conv_output_1, 点击 Add Output 添加输出


![](./images/021a.png)

同样选中第三个删除 Reshape上方的 Conv,修改 OUTPUTS 的名字为 Conv_output_2 , 点击 Add Output 添加输出


![](./images/022a.png)

最后点击左上角 Download 下载模型,修改好的模型可以在 modified_onnx 文件夹中找到,修改好的模型名字为 modified_best.onnx
![](./images/023.png)

### 六、打包训练图片
进入数据集的图片目录,使用以下命令打包图片为,注意文件的扩展名是 .tar

```bash
cd  ~/m3axpi/datasets/rubbish/images/
tar -cvf rubbish_1000.tar *.jpg
```

![](./images/030.png)

创建 dataset 目录,使用以下命令把压缩包 rubbish_1000.tar 移动到 ~/dataset 目录
```bash
mkdir -p ~/m3axpi/dataset
mv ~/m3axpi/datasets/rubbish/images/rubbish_1000.tar ~/m3axpi/dataset
```
![](./images/031.png)

### 搭建模型转换环境
onnx 模型需要转换为 joint 模型才能在 m3axpi 运行,所以需要使用 pulsar 模型转换工具。注意 pb、tflite、weights、paddle 等模型,需要先转换为 onnx 模型才能使用 pulsar 模型转换工具

使用以下命令拉取带有模型转换工具的容器,没有安装docker请自行安装

```bash
docker pull sipeed/pulsar:0.6.1.20
```
![](./images/032.png)

使用以下命令进入容器,如果需要保留该容器,请删除 --rm 这个参数。注意一定要设置共享内存,将 m3axpi 工作目录挂载到容器的 data 目录

```bash
cd ~/m3axpi
docker run -it --net host --rm --shm-size 16g -v $PWD:/data sipeed/pulsar
```

![](./images/033.png)

如果有 Nvidia GPU 环境,可以使用以下命令,使用带有GPU的容器,可以加快模型转换的速度

```bash
cd ~/m3axpi
docker run -it --net host --rm --gpus all --shm-size 16g -v $PWD:/data sipeed/pulsar
```

![](./images/034.png)

在工作目录下创建 config 和 onnx 文件夹。

```bash
cd ~/m3axpi
mkdir config onnx
```
![](./images/034a.png)

在 config 下创建一份命名为 yolov5s_rubbish.prototxt 的文件,复制以下内容到文件,注意修改文件中 rubbish_1000.tar 的路径
```
# my_config.prototxt

# 基本配置参数:输入输出
input_type: INPUT_TYPE_ONNX

output_type: OUTPUT_TYPE_JOINT

# 选择硬件平台
target_hardware: TARGET_HARDWARE_AX620

# CPU 后端选择,默认采用 AXE
cpu_backend_settings {
    onnx_setting {
        mode: DISABLED
    }
    axe_setting {
        mode: ENABLED
        axe_param {
            optimize_slim_model: true
        }
    }
}

# input
src_input_tensors {
    color_space: TENSOR_COLOR_SPACE_RGB
}

dst_input_tensors {
    color_space: TENSOR_COLOR_SPACE_RGB
}

# neuwizard 工具的配置参数
neuwizard_conf {
    operator_conf {
        input_conf_items {
            attributes {
                input_modifications {
                    affine_preprocess {
                        slope: 1
                        slope_divisor: 255
                        bias: 0
                    }
                }
                # input_modifications {
                #     input_normalization {
                #         mean: [0,0,0]
                #         std: [0.5,0.5,0.5]
                #     }
                # }
            }
        }
    }
    dataset_conf_calibration {
        path: "/data/dataset/rubbish_1000.tar" # 数据集图片的 tar 包,用于编译过程中对模型校准
        type: DATASET_TYPE_TAR         # 数据集类型:tar 包
        size: 256                      # 编译过程中校准所需要的实际数据个数为 256
    }
    dataset_conf_error_measurement {
        path: "/data/dataset/rubbish_1000.tar" # 用于编译过程中对分
        type: DATASET_TYPE_TAR
        size: 4                        # 对分过程所需实际数据个数为 4
        batch_size: 1
    }
    
}

dst_output_tensors {
    tensor_layout:NHWC
}

# pulsar compiler 的配置参数
pulsar_conf {
    ax620_virtual_npu: AX620_VIRTUAL_NPU_MODE_111
    batch_size: 1
    debug : false
}
```
![](./images/035.png)

移动编辑好的模型文件 best_cut.onnx 或 modified_best.onnx 到 onnx 目录,使用以下命令进行模型转换:(注意修改模型文件的名字改为自己的模型名字)

```bash
pulsar build --input onnx/best_cut.onnx --output yolov5s_rubbish.joint --config config/yolov5s_rubbish.prototxt --output_config yolov5s_rubbish.prototxt
```
开始转换
![](./images/036.png)

转换时间较长,请耐心等待

![](./images/037.png)

转换完成

![](./images/038.png)

可以在工作目录下找到转换后的模型 yolov5s_rubbish.joint

![](./images/039.png)