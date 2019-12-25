# YOLO v3 Training Automation API for Linux

This repository allows you to get started with training a state-of-the-art Deep Learning model with little to no configuration needed!  You provide your labeled dataset and you can start the training right away and monitor it in many different ways like TensorBoard or a custom REST API and GUI. Training with YOLOv3 has never been so easy.

## Prerequisites
- Ubuntu 18.04 [16.04 could work, but not tested]
- Install dependencies:
```bash
chmod +x scripts/install_dependencies.sh && source scripts/install_dependencies.sh
```
- Install docker:
```bash
chmod +x scripts/install_docker.sh && source scripts/install_docker.sh
```

##### Usage with NVIDIA Docker for GPU training
- Install NVIDIA Drivers and NVIDIA Docker for GPU training by following the [official docs](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

## Prepare Docker images
Once your environment is ready, you can prepare the docker images needed.
The environment is dockerized to run on GPU or CPU.

##### Include GPU usage
```bash
sudo docker build -f docker/Dockerfile -t darknet_yolo_gpu:1 --build-arg GPU=1 --build-arg CUDNN=1 --build-arg CUDNN_HALF=0 --build-arg OPENCV=1 .
```

##### Include CPU-only usage
```bash
sudo docker build -f docker/Dockerfile -t darknet_yolo_cpu:1 --build-arg GPU=0 --build-arg CUDNN=0 --build-arg CUDNN_HALF=0 --build-arg OPENCV=1 .
```

## Prepare your dataset
We provided a `sample_dataset` to show how your data should be structured in order to start the training seemlesly.
The `train_config.json` file found in `sample_dataset` is a copy of the template `config/train_config.json.template` with needed modifications.  The template can as well be copied as is while making sure to remove the '.template' from the name.
You can also provide your own `train.txt` and `test.txt` to specify which images will be used for training and which ones are for testing.  If not provided, the dataset will be split according to the `data/train_ratio` (by default 80% train 20% test)

## Start the training
##### Run training including GPU usage
```bash
chmod +x *.sh && ./run_docker_linux_gpu.sh
```

##### Run training including CPU-only usage
```bash
chmod +x *.sh && ./run_docker_linux_cpu.sh
```

The script will ask for 2 main inputs:
- The **absolute** path for the dataset
- The name of the container to be run (which will be also a prefix for the training output)
Once given, the training will start and you can stop it at any time by pressing CTRL+C inside the open terminal.
Closing the terminal will result in stopping the running container.

### Training output
Inside `trainings` you can find a folder with the naming convention `<container_name>_<training_start_time>`.
For example, it can be `dogs-dataset_20191110_14:21:41`. Inside this folder you will have the following structure:
```
dogs-dataset_20191110_14:21:41
├── config
│   ├── obj.data
│   ├── obj.names
│   └── yolov3.cfg
├── test.txt
├── train.txt
├── weights
│   ├── initial.weights
│   ├── yolov3_10000.weights
│   ├── yolov3_1000.weights
│   ├── yolov3_2000.weights
│   ├── yolov3_3000.weights
│   ├── yolov3_4000.weights
│   ├── yolov3_5000.weights
│   ├── yolov3_6000.weights
│   ├── yolov3_7000.weights
│   ├── yolov3_8000.weights
│   ├── yolov3_9000.weights
│   ├── yolov3_best.weights
│   └── yolov3_last.weights
├── yolo_events.log
└── yolo_events.log.1
```
That shows the _.cfg_ file and all weights used for the training along with all checkpoints
as well as the common YOLO related log output inside the `yolo_events` files.

## Monitoring the training
There are 3 ways to monitor the training process.

#### Custom API
A custom REST API including Swagger API will be started during training process. Thus, you can read the YOLO output log in a structured JSON format as well as test custom images on the latest saved weights. This can be accessed through port _8000_ (or a custom port you can set inside `training/custom_api/port`).

#### Tensorboard
Loss and mAP can be visualized through Tensorboard.
The interface can be accessed on port _6006_ (or a custom port you can set inside `training/tensorboard/port`).
![](tensorboard.png)

#### AlexeyAB provided web_ui
This can be enabled by setting `training/web_ui/enable` to `true` in the `train_config.json` you provide during the training.
It can later on be access through port _8090_ (or a custom port you can set inside `training/web_ui/port`) and looks like the following:
![Training monitoring - web_ui](https://camo.githubusercontent.com/d60dfdba6c007a5df888747c2c03664c91c12c1e/68747470733a2f2f6873746f2e6f72672f776562742f79642f766c2f61672f7964766c616775746f66327a636e6a6f64737467726f656e3861632e6a706567)

## Training config (meaning)
An explanation of the different fields can be found in the JSON schema of the provided config, which can be found at `config/train_config_schema.json`
Some of the elements are specific to YOLO itself - like saturation, hue, rotation, max_batches and so on.
Those are greatly explained by AlexeyAB in [his Darknet fork](https://github.com/AlexeyAB/darknet).

## Preparing weights
Default yolo weights are provided on the [official website](https://pjreddie.com/darknet/yolo/).
To download the different flavors, please use the following commands:
Change your current working directory to be inside the repo.
##### yolov3.weights
```bash
wget https://pjreddie.com/media/files/yolov3.weights -P config/darknet/yolo_default_weights
```
##### yolov3-tiny.weights
```bash
wget https://pjreddie.com/media/files/yolov3-tiny.weights -P config/darknet/yolo_default_weights
```

## Known Issues
Issue related to darknet itself can be filed in [the correct repo](https://github.com/AlexeyAB/darknet).  We did not make any changes to the darknet code itself.
- If you chose to build with GPU but in the configuration file did not provide "gpus" field, the training will run on gpu 0 by default
- If during training you see nan values for avg (loss) field - then training goes wrong, but if nan is in some other lines - then training goes well.
- If error Out of memory occurs then you should try increasing subdivisions to 16, 32 or 64 or have a smaller image size.
- If training finishes immediately without any error you should decrease batch size and subdivisions.