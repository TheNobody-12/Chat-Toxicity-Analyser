FROM python:3.8

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

COPY . . 

EXPOSE 80

CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
