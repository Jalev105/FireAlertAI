#!/usr/bin/python3
#this script runs the FireAlert AI model

import os
import sys
import cv2
import jetson_inference
import jetson_utils
import argparse
import pathlib

model = "~/FireAlertAI/models/FA2.onnx"
labels = "~/FireAlertAI/models/labels.txt"
version = "2.0.1"

parser = argparse.ArgumentParser(prog=f"Fire Alert AI v{version}")
parser.add_argument("--input", type=str, default="", help="Path to image or video file to submit to the model for detection. Not needed for livestream mode")
parser.add_argument("--output", type=str, default="", help="Path to save outputted image or video file. Not a required field")
parser.add_argument("--threshold", type=float, default=0.3, help="Confidence level in order to show prediction, default is 0.3 (30 percent)")
parser.add_argument("--livestream", action="store_true", help="Choose to livestream webcam (/dev/video0) to localhost:8554")
parser.add_argument("--test", action="store_true", help="input parameter can be shortened to file name (inside of Pictures folder) and output can be ommited")
parser.add_argument("-v", "--version", action="store_true", help="Shows the version number of the script")
opt = parser.parse_args()

if(opt.version):
    print(version)
    sys.exit(0)

if(opt.livestream):
    net = jetson_inference.detectNet(network="ssd-mobilenet-v1", model=os.path.expanduser(model), labels=os.path.expanduser(labels), input_blob="input_0", output_cvg="scores", output_bbox="boxes", threshold=opt.threshold)
    net.SetClusteringThreshold(0.4)
    input_stream = jetson_utils.videoSource("/dev/video0")
    output_stream = jetson_utils.videoOutput(
        "webrtc://@:8554/output",
        argv=["--headless", "--rtc-server", "--http-server"]
    )
    print("Waiting for connection")

    while True:
        img = input_stream.Capture()
        if img is None:
            continue

        detections = net.Detect(img)

        #convert image to numpy
        img_cv2 = jetson_utils.cudaToNumpy(img)

        for detection in detections:
            left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
            class_desc = net.GetClassDesc(detection.ClassID)
            conf = detection.Confidence

            #draw rectangle box
            cv2.rectangle(img_cv2, (left, top), (right, bottom), (255, 255, 0), 2)

            #add label text
            label = f"{class_desc}: {conf:.2f}"
            cv2.putText(img_cv2, label, (left + 5, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)

        #convert image back
        img_output = jetson_utils.cudaFromNumpy(img_cv2)

        output_stream.Render(img_output)

else:

    net = jetson_inference.detectNet(network="ssd-mobilenet-v1", model=os.path.expanduser(model), labels=os.path.expanduser(labels), input_blob="input_0", output_cvg="scores", output_bbox="boxes", threshold=opt.threshold)
    net.SetClusteringThreshold(0.4)
    print(pathlib.Path(opt.input).suffix)
    
    if(pathlib.Path(opt.input).suffix  in (".png", ".jpg", ".jpeg")):
        if(opt.test):
            filePathIn = os.path.expanduser("~/Pictures/") + opt.input
            filePathOut = os.path.expanduser("~/Pictures/") + pathlib.Path(opt.input).stem + "_" + pathlib.Path(model).stem + pathlib.Path(opt.input).suffix
        else:
            filePathIn = os.path.expanduser(opt.input)
            filePathOut = os.path.expanduser(opt.output)

        img = jetson_utils.loadImage(filePathIn)

        imgOld = cv2.imread(filePathIn)

        detections = net.Detect(img)
        box0 = []
        label0 = []
        prob0 = []
        print(f"detected {format(len(detections))} objects in image")
        for detection in detections:
            print("---------")
            print(net.GetClassDesc(detection.ClassID))
            print(detection)
            box0.append([detection.Left, detection.Top, detection.Right, detection.Bottom])
            label0.append(net.GetClassDesc(detection.ClassID))
            prob0.append(detection.Confidence)

        for i in range(len(box0)):
            box = box0[i]
            #cv2.rectangle(imgOld, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.rectangle(imgOld, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
            #label = f"""{labels[i]}: {probs[i]:.2f}"""
            label = f"{label0[i]}: {prob0[i]:.2f}"
            cv2.putText(imgOld, label,
                        (int(box[0]) + 20, int(box[1]) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        if(opt.output != "" or opt.test):
            cv2.imwrite(os.path.expanduser(filePathOut), imgOld)
            print(f"Image saved to {filePathOut}")
        else:
            print("Image not saved, --output not set")
    else:
        #input_stream = jetson_utils.videoSource(os.path.expanduser(opt.input))
        #output_stream = jetson_utils.videoOutput(os.path.expanduser(opt.output))
        print(os.path.expanduser("~/Pictures/") + pathlib.Path(opt.input).stem + "_" + pathlib.Path(model).stem + pathlib.Path(opt.input).suffix)
        if(opt.test):
            input_stream = jetson_utils.videoSource(os.path.expanduser("~/Pictures/") + opt.input)
            output_stream = jetson_utils.videoSource(os.path.expanduser("~/Pictures/") + pathlib.Path(opt.input).stem + "_" + pathlib.Path(model).stem + pathlib.Path(opt.input).suffix)
        else:
            input_stream = jetson_utils.videoSource(opt.input)
            output_stream = jetson_utils.videoOutput(opt.output)

        while True:
            img = input_stream.Capture()
            if img is None:
                continue

            detections = net.Detect(img)

            #img to numpy
            img_cv2 = jetson_utils.cudaToNumpy(img)

            for detection in detections:
                left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
                class_desc = net.GetClassDesc(detection.ClassID)
                conf = detection.Confidence

                #draw bounding box
                cv2.rectangle(img_cv2, (left, top), (right, bottom), (255, 255, 0), 2)

                #add label
                label = f"{class_desc}: {conf:.2f}"
                cv2.putText(img_cv2, label, (left + 5, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)

            #convert image back
            img_output = jetson_utils.cudaFromNumpy(img_cv2)

            output_stream.Render(img_output)