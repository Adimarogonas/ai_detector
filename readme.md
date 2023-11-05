Welcome to Huxli's Open Source AI Detector!

quick tour:
    - Detector.py - where we run our model and predictions, calculate perplexity   
    - gather_data.py - where we prepare our dataset for model training
    - train_model.py - where we actually run our model
    - runPrediction.py - where you can actually use this data
    - model-5.pkl - our serialized Model, you can replace this with your own!
    - results_1.csv - relevant data for training a model
    - calculated.csv - contains our real and fake passages. values here are outdated and it is not recommended to use them with provided models(will be very         inaccurate)

Getting Started:

1. Install dependences `pip3 install -r requirements.txt`
2. change text in runPredictions.py to text you want to test
3. run `python3 runPrediction.py`

System Requirements:
    - At least 3gb of disk space
    - at least 10gb of ram
    - While this runs on CPU a GPU is recommended
    - python
    - torch was built using a CPU version for linux for python 3.10. if you use a different version of python or operating system you will have to change     requirements.txt
    - A virtual environment is recommended for dependencies
    - Installing dependencies may be better with Anaconda 
