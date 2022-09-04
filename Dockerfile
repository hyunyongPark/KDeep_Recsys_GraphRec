FROM python:3.8

EXPOSE 8501
#/tcp

RUN apt-get update

RUN pip install --upgrade pip

# RUN pip install --upgrade cython

# RUN pip install --upgrade numpy

RUN mkdir -p /opt/graphrec

WORKDIR /opt/graphrec

COPY . .

COPY requirements.txt .

COPY GraphRec-kfashion_Inference.py .

COPY GraphRec-kfashion_training.py .

COPY ./kdeepfashion /kdeepfashion

COPY ./model_kfashion_add_externel /model_kfashion_add_externel

RUN pip install -r requirements.txt

CMD ["python", "./GraphRec-kfashion_Inference.py"]