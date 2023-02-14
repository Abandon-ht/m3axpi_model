</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

# [m3axpi] YOLOv8 train and convert model guide
### Introduction
YOLOv8 is a 2D real-time object detection model, open sourced by <a href="https://ultralytics.com"> Ultralytics </a>on January 10, 2023. Including multiple models. For instance yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov8x6. YOLOv8 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection, image segmentation and image classification tasks.

![](./images/index.jpg)

This tutorial explains the model training of YOLOv8 and deploy with Sipeed's development board m3axpi, Learn about the product at https://wiki.sipeed.com/en/m3axpi This is the actual shooting effect.

https://user-images.githubusercontent.com/32978053/216542487-0d17a9e4-ca81-4e67-a087-4ecf18b34feb.mp4

### 0. Get Started

First create a working directory, all the following operations are in this directory.

Right-click to open the terminal and execute the following command:
```bash
cd ~
mkdir m3axpi && cd m3axpi
```

### 1. Prepare dataset

How to make a object detection dataset, please refer to hyperlinks, This tutorial uses the marked "garbage detection" dataset to explain the whole process. This dataset can be obtained in the following three ways:

1. Download using browser

Github: https://github.com/Abandon-ht/coco_rubbish_dataset/archive/refs/heads/main.zip

Unzip the downloaded dataset to the datasets folder, and rename its folder to rubbish

2. Pull repository

```bash
mkdir datasets && cd datasets
git clone https://github.com/Abandon-ht/coco_rubbish_dataset.git rubbish
```
3. Download using command in terminal

```bash
mkdir datasets && cd datasets
wget https://github.com/Abandon-ht/coco_rubbish_dataset/archive/refs/heads/main.zip
unzip coco_rubbish_dataset-main.zip
mv coco_rubbish_dataset-main rubbish
```

All three methods can get the following 2 folders and 3 files

![](./images/002.png)

### 2. Pull the yolov8 repository

In the m3axpi directory (not in the datasets directory), pull the yolov8 repository

```bash
cd ~/m3axpi
git clone https://github.com/ultralytics/ultralytics.git  # clone
cd ultralytics
pip install -r requirements.txt  # install
pip install -e '.[dev]'  # develop
```

![](./images/003.png)

The yolov8 directory is shown in the picture:

![](./images/004.png)

### 3. Train yolov8 models

Switch to the working directory of ultralytics, copy coco.yaml under the data folderl, and rename it to rubbish.yaml

```bash
cp ultralytics/yolo/data/datasets/coco.yaml ultralytics/yolo/data/datasets/rubbish.yaml
```

Modify the path and classes name of the garbage classification dataset according to the picture

```bash
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

After the modification, train the yolov8s model with the following command

```bash
yolo task=detect mode=train model=yolov8s.pt data=rubbish.yaml batch=-1 epochs=20
```

![](./images/006.png)

The dataset is loaded successfully and the model training starts. If not loaded successfully, please check the dataset path

![](./images/007.png)

After the training is completed, you can view the training log under the ./runs/detect/train/ folder

PR_curve.png is the mAP_0.5 curve

![](./images/PR_curve.png)

results.png is all curves

![](./images/results.png)

### 4. Model Inference and Export

You can use the following command to predict the picture, note that you need to modify the path of the picture and the model to your own path

```bash
yolo task=detect mode=predict model=./runs/detect/train/weights/best.pt source=../datasets/rubbish/images/IMG_20210311_213716.jpg save
```

![](./images/008.png)

You can see the predicted images in the runs/detect/predict directory.

![](./images/009.jpg)

Modify ultralytics/nn/modules.py
class Detect(nn.Module):

```python
    def forward(self, x):
        shape = x[0].shape  # BCHW
        dfl = []
        cls = []
        cls_idx = []
        for i in range(self.nl):
            if self.export:
                dfl.append(self.cv2[i](x[i]))
                xi = self.cv3[i](x[i])
                cls.append(xi)
                cls_idx.append(torch.argmax(xi, dim=1, keepdim=True))
            else:
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        if self.export:
            return dfl, cls, cls_idx
        else:
            box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = torch.cat((dbox, cls.sigmoid()), 1)
            return y if self.export else (y, x)
```

Modify ultralytics/yolo/engine/exporter.py
class Exporter:

```python
        # self.output_shape = tuple(y.shape) if isinstance(y, torch.Tensor) else tuple(tuple(x.shape) for x in y)
        # LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with input shape {tuple(im.shape)} BCHW and "
        #             f"output shape(s) {self.output_shape} ({file_size(file):.1f} MB)")
```

Use the following command to export the onnx model, pay attention to add the parameter opset=11

```bash
yolo task=detect mode=export model=./runs/detect/train/weights/best.pt format=onnx opset=11
```

![](./images/010.png)

The exported onnx model is in the runs/detect/train/weights directory

![](./images/010a.png)

Enter netron.app in the browser address bar, Open the best.onnx file exported in the previous step, Check out the model structure of yolov8.

The input name of the model is 'images'

![](./images/011a.png)

Using the model exported before torch1.13.0, the last three convolution (Conv) outputs end with dfl、cls、cls_index

The first output matrix are 1 * 64 * 80 * 80, 1 * 16 * 80 * 80, 1 * 1 * 80 * 80

![](./images/014a.png)

The second output matrix are 1 * 64 * 80 * 80, 1 * 16 * 80 * 80, 1 * 1 * 80 * 80

![](./images/015a.png)

The third output matrix are 1 * 64 * 40 * 40, 1 * 16 * 40 * 40, 1 * 1 * 40 * 40

![](./images/016a.png)

### 5. Pack training pictures

Enter the image directory of the dataset, use the following command to package the image as rubbish_1000.tar, note that the extension of the file is .tar

```bash
cd  ~/m3axpi/datasets/rubbish/images/
tar -cvf rubbish_1000.tar *.jpg
```

![](./images/030.png)

Create a dataset directory, and use the following command to move the compressed package rubbish_1000.tar to the ~/dataset directory

```bash
mkdir -p ~/m3axpi/dataset
mv ~/m3axpi/datasets/rubbish/images/rubbish_1000.tar ~/m3axpi/dataset
```

![](./images/031.png)

### 6. Create a model conversion environment

The onnx model needs to be converted to a joint model to run on m3axpi, so the pulsar model conversion tool needs to be used. Note that pb, tflite, weights, paddle and other models need to be converted to onnx models before using the pulsar model conversion tool

Use the following command to pull the container with the model conversion tool, if you have not installed docker, please install it first

```bash
docker pull sipeed/pulsar:0.6.1.20
```

![](./images/032.png)

Use the following command to enter the container. If you need to keep the container, please delete the '--rm' parameter. Be sure to set up shared memory and mount the m3axpi working directory to the 'data' directory of the container

```bash
cd ~/m3axpi
docker run -it --net host --rm --shm-size 16g -v $PWD:/data sipeed/pulsar
```

![](./images/033.png)

If you have an Nvidia GPU environment, you can use the following command to use a container with GPU to speed up the model conversion

```bash
cd ~/m3axpi
docker run -it --net host --rm --gpus all --shm-size 16g -v $PWD:/data sipeed/pulsar
```

![](./images/034.png)

Create 'config' and 'onnx' folders in the working directory.

```bash
cd ~/m3axpi
mkdir config onnx
```
![](./images/034a.png)

Create a file named yolov8s_rubbish.prototxt under config, copy the following content to the file, and pay attention to modify the path of rubbish_1000.tar in the file
```
# my_config.prototxt

# Basic configuration parameters: input and output
input_type: INPUT_TYPE_ONNX

output_type: OUTPUT_TYPE_JOINT

# Select the hardware platform
target_hardware: TARGET_HARDWARE_AX620

# CPU backend selection, default AX
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

# Configuration parameters for the neuwizard tool
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
        path: "/data/dataset/rubbish_1000.tar" # The tar package of the dataset image, used to calibrate the model during compilation
        type: DATASET_TYPE_TAR         # Dataset type: tar package
        size: 256                      # The actual number of data required for calibration during compilation is 256
    }
    dataset_conf_error_measurement {
        path: "/data/dataset/rubbish_1000.tar" # Used for bisection during compilation
        type: DATASET_TYPE_TAR
        size: 4                        # The actual number of data required for the bisection process is 4
        batch_size: 1
    }

}

dst_output_tensors {
    tensor_layout:NHWC
}

# Configuration parameters for pulsar compiler
pulsar_conf {
    ax620_virtual_npu: AX620_VIRTUAL_NPU_MODE_111
    batch_size: 1
    debug : false
}
```
![](./images/035.png)

Move the export model file best.onnx to the onnx directory, and use the following command to convert the model: (note that the name of the modified model file is changed to your own model name)

```bash
pulsar build --input onnx/best.onnx --output yolov8s_rubbish.joint --config config/yolov8s_rubbish.prototxt --output_config yolov8s_rubbish.prototxt
```
start converting
![](./images/036.png)

The conversion time is long, please wait patiently

![](./images/037.png)

conversion complete

![](./images/038.png)

The converted model yolov8s_rubbish.joint can be found in the working directory

![](./images/039.png)

### 8. Deployment

Please refer to https://github.com/AXERA-TECH/ax-samples/blob/main/README_EN.md
(to be continued)
