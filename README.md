# ISA course: ML/DL deployment - Image data type

Example of image classification inference using a CNN model. You can upload your images either by dragging them into the chat area or clicking on "Upload image" button. Then just click on the Send button and see what your model predicted.

Author(s):
- Matej Volansky (2024)
- as a part of the team work preparation in 2023/2024

Docker image size: `1.25 GB`

## Setup
First train your model and save the `.pt` state dictionary after training (considering you're using PyTorch). Update the `model/model.py` with your model. 

## Run
To use this inference, just run 

```
docker compose up
```
After successful build, your server will be available at `http://localhost:8080/` 


For manual running without docker you have to create a python virtual environment.

```
python -m venv venv

source venv/bin/activate          # on Linux based distros
source venv\Scripts\activate.ps1  # on Windows (Powershell)
source venv\Scripts\activate.bat  # on Windows (cmd)
```

To install the PyTorch CPU version run
```
pip install torch==2.6.0 torchvision==0.21.0
```
After that go ahead and run
```
pip install -r requirements.txt
```
