#Image: avlavelle/starfish

FROM python:3.11

RUN pip install Flask==3.0
RUN pip install pandas==2.2.1
RUN pip install scikit-learn==1.4.1.post1
RUN pip install xgboost==2.0.3
RUN pip install requests==2.31.0

COPY models /models
COPY api.py /api.py
COPY testdata.csv /testdata.csv

CMD ["python", "api.py"]
