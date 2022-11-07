# Multimodal deception detection in trial videos using intermediate fusion and interpretation methods
Intermediate fusion approach for [Real-life trial deception detection dataset](https://web.eecs.umich.edu/~mihalcea/downloads.html#RealLifeDeception). The task is binary classification (truth:0 , lie: 1), using the Gated Multimodal Unit ([GMU](https://arxiv.org/abs/1702.01992)) and Layer-wise Relevance Propagation ([LRP](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)) to explore learned patterns.

All extracted features should be placed in data folder. We use the following toolkits and pre-trained networks:
- [OpenFace](https://ieeexplore.ieee.org/abstract/document/7477553) for automatic FAU features
- Face embeddings with this [keras implementation](https://github.com/rcmalli/keras-vggface) of [VGGFace2](https://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf) (see also [this repo](https://github.com/WeidiXie/Keras-VGGFace2-ResNet50))
- [OpenSmile](https://dl.acm.org/doi/abs/10.1145/2502081.2502224) for audio statistics
- Audio embeddings extracted with [VGGVox](https://www.robots.ox.ac.uk/~vgg/publications/2019/Nagrani19/nagrani19.pdf)

Extracted features to run the example in train.py are available as csv or embeddings [here](https://drive.google.com/drive/folders/1hTiR3Xhz4fjF5Fa7hrlYc6jDxOncyPF3?usp=share_link)
