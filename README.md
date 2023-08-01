# nvidia-project
# pill finder

This app determines the which of the three main pain reliever pills are which and helps differentiate between them.

![The 3 pill bottles](https://images.thestarimages.com/RaEg5D2DY6mCH6g_8LAgsgnhB78=/1200x800/smart/filters:cb(2700061000)/https://www.thestar.com/content/dam/thestar/life/health_wellness/2017/04/03/advil-tylenol-and-other-over-the-counter-painkillers-come-with-risks/rlpainmed01jpg.jpg)
## The Algorithm

The program uses a retrain version of resnet 18 to differentiate between three pill bottles, and recommends the appropriate pill for a specific symptom and also provides the recommended dosage. 

## Running this project

1. git clone https://github.com/jamiefinn07/pill-ai.git
2. read app information, preinstall jetson inference
3. Grab pill bottle
4. scan through camera
5. take apropiate dosage and pill according to information
6. run this code python3 my-recognition.py /dev/video0 rtp://<Your Ip>:1234

[View a video explanation here](https://youtu.be/47oyuavFB8Q)
