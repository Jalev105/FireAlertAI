# **FireAlert AI README**

[View Video Demo](https://www.youtube.com/watch?v=D0USiicJHjY)

FireAlert AI is a program that can detect smoke and fire using an AI object detection model. It contains documentation on how to run the pretrained models that come with the repository and how to train FireAlertAI models yourself on NVIDIA Jetson Orin Nano devices. This kind of system can provide early alerts about wildfires, potentially helping to reduce fire danger in remote locations 

(Jetson inference must be installed for this project to work)

The dataset is provided by Azimjon Akhtamov from Kaggle [(View Here - 6.84GB)](https://www.kaggle.com/datasets/azimjaan21/fire-and-smoke-dataset-object-detection-yolo/data)

Progress list
- [x] Decided on final project
- [x] Found dataset
- [x] Preprocessed dataset
- [x] Trained model
- [x] Tested model
- [x] Created python script to run model
- [x] Added livestream support
- [x] Added video support
- [x] Trained v2 of the model
- [x] Added web ui dashboard to use model from

## Demos:

| Before | After |
| ------ | ------ |
| ![01activities](https://github.com/user-attachments/assets/85dcf169-3764-45ac-ab2a-d61b60918eba) | ![01activities_FA2](https://github.com/user-attachments/assets/ad804dee-7982-47f1-be08-22b0cc94170d) |
| ![imFire](https://github.com/user-attachments/assets/f279126b-fa17-4080-a3e6-12015fdc5bfc) | ![imFire_FA2](https://github.com/user-attachments/assets/e1313baa-08a9-45f4-98f6-a7d865d3f805) |
| ![3000](https://github.com/user-attachments/assets/139a44df-0dd1-43e0-939f-f071a0becc23) | ![3000_FA2](https://github.com/user-attachments/assets/b5867b28-655d-4f61-bc72-dada0d5a275f) |








## Usage:

### Download General Data:

- Clone FireAlert AI repository for instructions and other code

### Train Model:

- cd into data directory with `cd ~/jetson-inference/python/training/detection/ssd/data/`
- Download dataset with: 
```
#!/bin/bash
curl -L -o fire-and-smoke-dataset-object-detection-yolo.zip\
  https://www.kaggle.com/api/v1/datasets/download/azimjaan21/fire-and-smoke-dataset-object-detection-yolo
```
- Unzip folder with `unzip fire-and-smoke-dataset-object-detection-yolo.zip`
- (Optional) Remove zipped file with `sudo rm -f fire-and-smoke-dataset-object-detection-yolo.zip`
- cd to FireAlertAI directory and run `python3 yoloToVOC.py` (change paths if needed)
- ~~At the moment, some files in the dataset cause errors when training, run `python3 removeFilesFromDataset.py` in order to remove them~~
- cd into data directory with `cd ~/jetson-inference/python/training/detection/ssd/data/`
- (Optional) Remove YOLO dataset with `sudo rm -rf fire_smoke`
- `cd ..` up into the ssd folder
- Start training with: 
```
python3 train_ssd.py --dataset-type=voc \
--data=data/fire_smoke_voc \
--model-dir=models/FA \
--num-epochs=195
```
**Note**: you can change the `--num-epochs` flag based on how much you want your model to train
- If you need to stop training for any reason, you can ctrl-c the script and restart from where you left off later with:
```
python3 train_ssd.py --dataset-type=voc \
--data=data/fire_smoke_voc \
--model-dir=models/FA \
--num-epochs=195 \
--resume=$HOME/jetson-inference/python/training/detection/ssd/models/FA/mb1-ssd-(your file here).pth
```
**Note**: It will say that you are starting at Epoch 0, but it will remember and use all of the saved data from the .pth file you selected

- Once finished, export the best model to the Fire Alert AI folder with `python3 onnx_export.py --model-dir=models/FA --output=$HOME/FireAlertAI/FA.onnx` (remember the saved path for the step below)

### Testing the Model:

cd to Fire Alert AI folder and run `python3 runFireAlert.py --input=(input image/video here) --output=(output image/video here)` (replacing the input and output paths with the file you want to classify and save)

**Note**: This script runs the pretrained model that comes with the repository (FA2.onnx), if you want to run the model that you created in the section above, edit the model variable in the script. Additionally, the models folder also comes with three other models (FA1.onnx, FA3.onnx, and FA4.onnx). These models do not work as accurately as FA2, but you are free to try them by changing the model path or using the web UI below.

### Other Options:

```
usage: python3 runFireAlert.py [-h] [--input INPUT] [--output OUTPUT] [--threshold THRESHOLD] [--livestream] [--test] [-v]

options:
  -h, --help                show this help message and exit
  --input INPUT             Path to image or video file to submit to the model for detection. Not needed for livestream mode
  --output OUTPUT           Path to save outputted image or video file. Not a required field
  --threshold THRESHOLD     Confidence level in order to show prediction, default is 0.3 (30 percent)
  --livestream              Choose to livestream webcam (/dev/video0) to localhost:8554
  --test                    input parameter can be shortened to file name (inside of Pictures folder) and output can be ommited
  -v, --version             Shows the version number of the script
```

### Web UI Dashboard

This version of FireAlertAI includes a custom web UI dashboard for making inferences on images. Models are selected via the dropdown in the menu, to use your own model move it (FA.onnx) to the models folder in FireAlertAI.

To run the dashboard:
- Start the backend by moving into the FireAlertAI/web folder and running `python3 FireAlertServer.py`
- Start a webserever in a separate terminal (also in the FireAlertAI/web folder) to run the html interface
  - A simple option is `python3 -m http.server 8000` (change the port as you wish)
  - Another one is the Apache HTTP server, requires extra work to install
- To make sure the system is working, open <http://localhost:8000> (change the port to what you started the server with) and check that the models list is not empty
- Upload an image and press the run detection button, the first time will be slow but after that it should go quicker

**Note:** After switching from one model to another, the system has to load different files and the next detection may go slowly
