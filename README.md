# Vanilla-GAN
<br/>

### Datasets
MNIST : 0 ~ 9까지의 손글씨 숫자 데이터로서 60,000개의 trainset 10,000개의 testset으로 구성

### Model
Vanilla-GAN
- Generator      : hidden layer(256, 512, 1024) + LeakyReLU + dropout, output layer(784) + Tanh
- Discriminator  : hidden layer(1024, 512, 256) + LeakyReLU + dropout, output layer(1) + Sigmoid
  
