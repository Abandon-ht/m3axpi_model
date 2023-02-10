</div>

<div align="center">

English | [简体中文](README.md)

</div>

# [m3axpi] YOLOv5 train and convert model guide
### Introduction
YOLOv5 is a 2D real-time object detection model, open sourced by <a href="https://ultralytics.com"> Ultralytics </a>on June 9, 2020. Including multiple models. For instance yolov5n, yolov5n6, yolov5s, yolov5s6, yolov5m, yolov5m6, yolov5l, yolov5l6, yolov5x, yolov5x6. YOLOv5 has many advantages. For instance fast training speed, little inference time, easy to train and deploy. The neural network structure of YOLOv5 can be divided into four parts intput, backbone, neck, head.

![](./images/index.jpg)

This tutorial explains the model training of YOLOv5 and deploy with Sipeed's development board m3axpi, Learn about the product at https://wiki.sipeed.com/en/m3axpi This is the actual shooting effect.

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

### 2. Pull the yolov5 repository

In the m3axpi directory (not in the datasets directory), pull the yolov5 repository

```bash
cd ~/m3axpi
git clone -b v7.0 https://github.com/ultralytics/yolov5.git  # clone
cd yolov5
pip install -r requirements.txt  # install
```

![](./images/003.png)

The yolov5 directory is shown in the picture:

![](./images/004.png)

### 3. Train yolov5 models

Switch to the working directory of yolov5, copy coco.yaml under the data folderl, and rename it to rubbish.yaml

```bash
cp data/coco.yaml data/rubbish.yaml
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

After the modification, train the yolov5s model with the following command

```bash
python train.py --data data/rubbish.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --batch-size -1 --epoch 20
```

![](./images/006.png)

The dataset is loaded successfully and the model training starts. If not loaded successfully, please check the dataset path

![](./images/007.png)

After the training is completed, you can view the training log under the ./runs/train/exp/ folder

PR_curve.png is the mAP_0.5 curve

![](./images/PR_curve.png)

results.png is all curves

![](./images/results.png)

### 4. Model Inference and Export

You can use the following command to predict the picture, note that you need to modify the path of the picture and the model to your own path

```bash
python detect.py --source ../datasets/rubbish/images/IMG_20210311_213716.jpg --weights ./runs/train/exp/weights/best.pt
```

![](./images/008.png)

You can see the predicted images in the runs/detect/exp directory.

![](./images/009.jpg)

Use the following command to export the onnx model, pay attention to add the parameter opset=11

![](./images/010.png)

```bash
python export.py --include onnx --opset 11 --weights./runs/train/exp/weights/best.pt
```

The exported onnx model is in the runs/train/exp/weights directory

![](./images/010a.png)

Enter netron.app in the browser address bar, Open the best.onnx file exported in the previous step, Check out the model structure of yolov5.

The input name of the model is 'images'

![](./images/011a.png)

Using the model exported before torch1.13.0, the last three convolution (Conv) outputs end with onnx::Reshape_329. The ending numbers of the three convolution (Conv) outputs are not the same. The numbers at the end of your exported model may not be the same as mine.

The output of the first Conv is onnx::Reshape_329

![](./images/012a.png)

The output of the second Conv is onnx::Reshape_367

![](./images/013a.png)

The output of the third Conv is onnx::Reshape_405

![](./images/014a.png)

### 5. Modify the ONNX model

Because the exported onnx model has post-processing, and the YOLOv5 model deployed on m3axpi is post-processing implemented through code. So you need to delete the post-processing part of the model. The following scripts can be used to modify the model. Note that if the model is exported with torch1.13.0 and later, please use method 2 described below to use a graphical method to modify the model

1. Modify ONNX model with python script

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
Create a new file named onnxcut.py and copy the above code

Use the following command, be sure to replace the three outputs of the model with the output of the last three convolutions (Conv) of your model

```bash
python onnxcut.py --onnx_input ./runs/train/exp/weights/best.onnx --onnx_output ./runs/train/exp/weights/best_cut.onnx --model_input images --model_output onnx::Reshape_329 onnx::Reshape_367 onnx::Reshape_405
```

![](./images/015.png)

The modified model is shown in the picture, and the three outputs are onnx::Reshape_329, onnx::Reshape_367, onnx::Reshape_405

Note: The output matrix I demonstrate here is 1 * 255 * 80 * 80, which are batch size(1), filters(255) = ( class numbers(80) + bbox(4)(x, y, h, w) + obj(1) ) * anchor numbers(3), h, w

In fact, the three output matrices of this rubbish classification model are 1 * ((16+4+1) * 3) * 80 * 80, 1 * ((16+4+1) * 3) * 40 * 40, 1 * ((16+4+1) * 3) * 20 * 20

![](./images/016a.png)

2. use a graphical method to modify the model

You can also modify the model using graphical methods, and use the following command to pull the source code repository

```bash
cd ~/m3axpi
git clone https://github.com/ZhangGe6/onnx-modifier.git
```

![](./images/016b.png)

The directory is as shown in the picture:

![](./images/017.png)

Enter the directory and use the following command to install the dependency package

```bash
cd onnx-modifier
pip install onnx flask
```
![](./images/017a.png)

Run the script with
```bash
python app.py
```
![](./images/017b.png)

Enter http://127.0.0.1:5000 in the browser address bar. Open the best.onnx file exported in the third step, and modify the model structure.

Select the Reshape after the last three convolutions (Conv) in turn, and click 'Delete With Children'

![](./images/018a.png)

Select the first one to delete the Conv above 'Reshape', change the name of OUTPUTS to 'Conv_output_0', and click 'Add Output' to add the output

![](./images/019a.png)

Select the second Reshape, click 'Delete With Children'

![](./images/020a.png)

Select the second one to delete the Conv above 'Reshape', change the name of OUTPUTS to 'Conv_output_1', click 'Add Output' to add the output

![](./images/021a.png)

Also select the third one to delete the Conv above 'Reshape', change the name of OUTPUTS to 'Conv_output_2', click 'Add Output' to add the output

![](./images/022a.png)

Finally, download the model. The modified model can be found in the modified_onnx folder. The modified model name is modified_best.onnx

![](./images/023.png)

### 6. Pack training pictures

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

### 7. Create a model conversion environment

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

Move the modified model file best_cut.onnx or modified_best.onnx to the onnx directory, and use the following command to convert the model: (note that the name of the modified model file is changed to your own model name)

```bash
pulsar build --input onnx/best_cut.onnx --output yolov5s_rubbish.joint --config config/yolov5s_rubbish.prototxt --output_config yolov5s_rubbish.prototxt
```
start converting
![](./images/036.png)

The conversion time is long, please wait patiently

![](./images/037.png)

conversion complete

![](./images/038.png)

The converted model yolov5s_rubbish.joint can be found in the working directory

![](./images/039.png)

### 8. Deployment

Please refer to https://github.com/Abandon-ht/ax-samples/edit/main/README_EN.md
(to be continued)
