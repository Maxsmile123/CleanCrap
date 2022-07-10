FROM continuumio/miniconda3

COPY . .

WORKDIR /CleanCrap

RUN conda create -n e2fgvi python=3.7 && echo "conda activate e2fgvi" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

RUN conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 opencv -c pytorch \
&& pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5/index.html \
&& conda install tensorboard matplotlib scikit-image==0.16.2 \
&& pip install tqdm \
&& apt-get update \
&& apt-get install ffmpeg libsm6 libxext6  -y \
&& rm -rf /var/lib/apt/lists/* \
&& pip install streamlit

EXPOSE 8501

CMD streamlit run app.py
