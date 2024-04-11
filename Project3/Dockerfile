#Image: aJimenez19037/HurricaneAPI

FROM python:3.11

RUN pip install tensorflow==2.15
RUN pip install Flask==3.0
Run pip install tensorflow_datasets --user 

COPY keras_models /keras_models
COPY api.py /api.py


CMD ["python", "api.py"]
