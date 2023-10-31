# Vanilla-GAN
<br/>

### Datasets
MNIST : 0 ~ 9까지의 손글씨 숫자 데이터로서 60,000개의 trainset 10,000개의 testset으로 구성
<p align="center">
  <img src="./result/MNIST.webp" width="300" height="200"/>
</p>

### Model
-------------------
Vanilla-GAN
- Generator      : hidden layer(256, 512, 1024) + LeakyReLU + dropout, output layer(784) + Tanh
- Discriminator  : hidden layer(1024, 512, 256) + LeakyReLU + dropout, output layer(1) + Sigmoid

### Train
-------------------
epochs : 100, learning rate : 0.0005, optimizer : Adam(Beta1, 2 = 0.5), Loss : MiniMax Loss
<br/>

### Result
-------------------
#### 1. D, G Loss
<p align="center">
  <img src="./result/result2.png" width="500" height="350"/>
</p>

#### 2. D, G Performance
<p align="center">
  <img src="./result/result1.png" width="500" height="350"/>
</p>

#### 3. Inference
<p align="center">
  <img src="./result/out.gif" width="350" height="350"/>
</p>

#### 4. Generated Image corresponding to change in z value
<p align="center">
  <img src="./result/out2.gif" width="440" height="350"/>
</p>
