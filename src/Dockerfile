FROM tensorflow/tensorflow:2.9.1-jupyter

RUN pip install tf-agents==0.13.0
RUN pip install dm-reverb==0.8.0

# https://github.com/tensorflow/agents/issues/762
RUN sed -i 's/training=False/training=False, mask=None/g' /usr/local/lib/python3.8/dist-packages/tf_agents/networks/value_network.py

WORKDIR "/tf"
RUN rm -rf tensorflow-tutorials
#COPY agents /tf/agents
#COPY environments /tf/environments
COPY ./ /tf/