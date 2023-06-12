FROM python:3.8
# copy the dependencies file to the working directory
COPY . /app
WORKDIR /app
# install dependencies
RUN pip install -r requirements.txt
# In order to access the app in container we need port. so we use expose it as port value
EXPOSE 5000
#  -bind is used to bind the port to the app and workers is used to handle the request and response  of the app
CMD python app.py

