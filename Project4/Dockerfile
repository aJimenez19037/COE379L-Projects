#Image: avlavelle/starfish

FROM python:3.11

RUN pip install Flask==3.0
RUN pip install pandas==2.2.1

COPY models /models
COPY api.py /api.py

CMD ["python", "api.py"]
