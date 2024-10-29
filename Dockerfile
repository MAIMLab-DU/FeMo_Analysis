FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt /code/
RUN pip install -r requirements.txt

COPY . /code/

RUN pip install .

CMD [ "python3" ]

