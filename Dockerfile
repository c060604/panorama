FROM c060604/opencv:0.0.1
MAINTAINER c060604 "c060604@gmail.com"
ADD . /code
WORKDIR /code
RUN pip3 install -r requirements.txt
CMD python3 panorama.py
