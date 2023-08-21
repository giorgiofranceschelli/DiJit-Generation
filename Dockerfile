FROM tensorflow/tensorflow:2.9.0-gpu

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "./main.py"]