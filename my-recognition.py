#!/usr/bin/python3
import sys
import jetson_inference
import jetson_utils
from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log
import time
import argparse
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
#parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
#opt = parser.parse_args()
#img = jetson_utils.loadImage(opt.filename)
#net = jetson.inference.imageNet(opt.network)
try:
    args = parser.parse_known_args()[0]
except:  
    print("")
    parser.print_help()
    
net = imageNet(model="resnet18.onnx", labels="labels.txt", 
                input_blob="input_0", output_blob="output_0")
# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
font = cudaFont()

while True:
    img = input.Capture()

    if img is None: # timeout
        continue
    class_idx, confidence = net.Classify(img)
    class_desc = net.GetClassDesc(class_idx)
    # print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))
    # print(f"{class_idx}")
    # print(f"{class_desc}")
    advil_facts = [
        "Advil is a brand name for the NSAID ibuprofen.",
        "It relieves pain, reduces inflammation, and lowers fever.",
        "Short-term use recommended; follow dosage instructions.",
        "Possible side effects: stomach irritation, dizziness, allergic reactions.",
        "Consult a doctor for specific health concerns or interactions with other medications.",
        "Pros: Effective pain and fever relief, anti-inflammatory properties.",
        "Cons: Stomach irritation, interactions with certain medications, potential kidney issues."
    ]
    aspirin_info = [
        "Active Ingredient: Aspirin contains acetylsalicylic acid.",
        "Uses: It provides pain relief, reduces inflammation, and lowers fever. It's effective for headaches, muscle aches, arthritis, and menstrual cramps.",
        "Blood Thinning: Aspirin has blood-thinning properties and may be prescribed to reduce the risk of blood clots, heart attacks, or strokes.",
        "Stomach Irritation: It can cause stomach irritation and may lead to ulcers or bleeding, especially with high doses or prolonged use.",
        "Reye's Syndrome: Avoid giving aspirin to children or teenagers with viral infections due to the risk of Reye's syndrome.",
        "Interactions: Aspirin may interact with other medications, so it's essential to inform healthcare providers about all medications being taken.",
        "Precautions: Caution is needed in cases of bleeding disorders, ulcers, asthma, or allergies to NSAIDs.",
        "Dosage: Follow recommended dosages and guidelines to avoid overdose.",
        "Aspirin for Heart Health: Consult a doctor before using aspirin for heart attack or stroke prevention.",
        "Pregnancy: Aspirin is generally not recommended during pregnancy, especially in the third trimester.",
        "Pros: Effective pain relief and anti-inflammatory properties, blood-thinning effects.",
        "Cons: Stomach irritation, risk of bleeding, avoid in children and certain conditions."
    ]
    tylenol_info = ["Active Ingredient: Tylenol contains acetaminophen.",
        "Pain Relief and Fever Reduction: Effective for mild to moderate pain and fever.",
        "Non-anti-inflammatory: Unlike Advil and aspirin, Tylenol doesn't reduce inflammation significantly.",
        "Dosage and Usage: Follow recommended dosages to avoid overdose, consult a healthcare professional.",
        "Liver Safety: Excessive use or alcohol combination can harm the liver.",
        "Interactions: Inform healthcare provider of all medications to prevent interactions.",
        "Caution for Certain Conditions: Use with caution if you have liver problems.",
        "Avoid in Children and Teens with Viral Infections: Due to Reye's syndrome risk.",
        "Overdose Concerns: Can cause severe liver damage or be fatal if overdosed.",
        "Use Responsibly: Follow recommended dosage, avoid interactions with other drugs or alcohol. Consult a healthcare professional for specific concerns."]

    # class_desc = "Tylenol"
    class_desc.strip()
    class_desc.replace(" ", "")
    if class_desc[0:2] == "ty":
        for item in tylenol_info:
            print(item)
    elif class_desc[0:2] == "as":
        for item in aspirin_info:
            print(item)
    elif class_desc[0:2] == "ad":
        for item in advil_facts:
            print(item)
    time.sleep(1)