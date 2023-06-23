# Introduction 
Social Media Platforms are a way of expressing thoughts and opinions. In some cases, these online chats or messages contain explicit language which may hurt the readers.
Mostly the comments having toxic nature are classified as Toxic, Severe Toxic, Threat, Insult and Identity Hate.The goal of this project was to make an application which can moderate chat messages and classify them into different categories. The application can be used by social media platforms to moderate the comments and take necessary actions. Currently as an initial stage, ML model is made using Keras TextVectorizer and Bidiretional LSTM. The model is trained on the dataset provided by Kaggle. The model is deployed on Flask and can be used to classify the comments. The model is trained on 159571 comments and tested on 63978 comments. The model is able to classify the comments with 97% accuracy.

# Getting Started
1.	Installation process
    - Clone this repository

    - Install  python version 3.9 or higher version

    - Install Virtual Environment using Pip in command line.
    ` pip install virtualenv`

    - Create a Virtual Environment in command line.
    `  python<version> -m venv <virtual-environment-name>`

    - Activate the Environment in command line.
    ` .\<virtual-environment-name>\Scripts\activate`

    - In activated environment, install all python dependencies.
    ` pip install -r requirements.txt`

    - Run code file using below command.
    ` python app.py`

2.	Software dependencies
    #### Frontend
    - HTML
    - CSS
    - JavaScript
    #### Backend:
    - Python
    - Flask
    #### ML Training
    - Google Colab
    - Keras
    - Tensorflow
    - Pandas
    - Numpy
    - Matplotlib
3.	Latest releases
    - 1.0.0
4.	API references
    - [Flask](https://flask.palletsprojects.com/en/2.0.x/)
    - [Keras](https://keras.io/)
    - [Tensorflow](https://www.tensorflow.org/)
    - [Pandas](https://pandas.pydata.org/)
    - [Numpy](https://numpy.org/)
    - [Matplotlib](https://matplotlib.org/)
    - [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
    - [Kaggle](https://www.kaggle.com/)
    - [Jupyter Notebook](https://jupyter.org/)
    - [Python](https://www.python.org/)

# Deployed API Link 
- [Toxic Comment Classifier](https://chatmode.techdomeaks.com/) 

# Build and Test

You can build and test the project using the following steps:
1.

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)