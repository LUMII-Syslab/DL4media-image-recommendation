FROM python:3.7-bullseye

RUN apt update
RUN apt install libtinfo5

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./code/model_dl.py /app/code/model_dl.py
RUN python ./app/code/model_dl.py

COPY ./code/get_torch.py /app/code/get_torch.py
RUN python ./app/code/get_torch.py

RUN wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -O ./app/model.pt
RUN wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz -O ./app/bpe_simple_vocab_16e6.txt.gz

COPY ./code/get_clip.py /app/code/get_clip.py
RUN pip install git+https://github.com/openai/CLIP.git
RUN python ./app/code/get_clip.py

#RUN pip install editdistance

#RUN pip install flair


COPY ./code/get_ner.py /app/code/get_ner.py
RUN python ./app/code/get_ner.py

COPY . /app
WORKDIR /app

# Expose the Flask port
EXPOSE 5000

CMD [ "python", "-u", "./main.py" ]
