# Generative Models

## Supervised Learning vs. Unsupervised Learning

**Supervised Learning**

<img src="https://1.bp.blogspot.com/-w8fK__8KmYQ/XZbzhrhexqI/AAAAAAAACac/HVw_rXHnF4wcI4sbWVCsVtcQ2b9tbOQrACLcBGAsYHQ/s640/%25EC%25BA%25A1%25EC%25B2%2598.JPG" width="50%">

 - 입력 데이터(x)에 대한 label(y)이 같이 제공된다
 - x -> y로 mapping 하는 과정을 학습한다
 - Classification, Object Detection, Semantic Segmentation, Image Captioning등에 사용

**Unsupervised Learning**

<img src="https://1.bp.blogspot.com/-axh7lJgoqAI/XZbz1vvWgeI/AAAAAAAACak/41E4GpEHrZIBd_k2cmzZ1zs4MfZ5Z3Z1ACLcBGAsYHQ/s640/%25EC%25BA%25A1%25EC%25B2%2598.JPG" width="50%">

- 레이블 없이 입력 데이터(x)만 제공된다
- 데이터의 **hidden structure**을 학습시키기 위해 수행한다
- 정답이 주어진 데이터만 학습할 수 있는 지도학습과 달리 사용할 수 잇는 데이터가 훨씬 많기에(Training Data is cheap) 전문가들은 인공지능 기술은 지도학습이 아닌 비지도학습이 선도할 것이라 전망하고 있음
- Clustering, Dimensionality Reduction, Feature Learning, Density Estimation 등에 사용
	- 차원 축소(Dimensionality Reduction) : 시각화를 위해 데이터셋을 2차원으로 변경하거나 이미지 데이터를 압축하는 경우가 있음
	- Autoencoders : 데이터를 효율적으로 나타내기 위하여(Feature Learning) 고차원을 저차원으로 차원 축소하는 방법
		-  x 데이터에서 특징 z로 변환하는 매핑 함수의 역할

## Generative Models
<img src="https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/assets/gedl_0101.png">

- Training Data ~ p~data~(x), Generated Samples ~ p~model~(x)
- p~model~(x)가 p~data~(x)에 유사하도록 학습시킴
- 분포 추정(Density Estimation)를 하는 것이 핵심 문제인데, 이에는 2가지 방법이 있다
	- Explicit Density Estimation : 생성모델 p~model~(x)를 명시적으로 나타내는 방법
	- Implicit Density Estimation : 생성모델 p~model~(x)를 정의하지 않고 sample을 얻어내는 방법

<img src="https://qjjnh3a9hpo1nukrg1fwoh71-wpengine.netdna-ssl.com/wp-content/uploads/2019/03/ai_research_gan_1600px_web-1080x540.jpg" width="50%">

- 시계열 데이터에 대해 시뮬레이션과 planning이 가능하며 이를 강화학습에 적용 가능
- general feature과 같이 유용한 잠재적 특성을 얻을 수 있다

## Taxonomy of Generative Models

<img src="https://i0.wp.com/christineai.blog/wp-content/uploads/2019/12/Screen-Shot-2019-12-28-at-1.59.52-AM.png?resize=769%2C368&ssl=1">

## PixelRNN and PixelCNN

<img src="https://user-images.githubusercontent.com/59776953/119317157-bf9efb80-bcb2-11eb-9cdc-e3f5e0e45ce0.png">

- 이미지 x에 대한 likelihood p(x)를 모델링 한 식
-  chain rule을 통하여 likelihood를 1-d distribution의 곱으로 나타냄
- training data의 likelihood를 최대화
- 이전의 pixel들을 모두 사용하는데 pixel의 순서를 정의해야 함

**PixelRNN**
<img src="https://user-images.githubusercontent.com/59776953/119318185-f295bf00-bcb3-11eb-939f-429a6ba898d4.jpg" width="50%">

- 코너에서부터 화살표를 따라 pixel 생성
- 이전 pixel에 대한 의존성을 바탕으로 RNN(LSTM)을 이용
- 순차적 생성이 느리다는 단점을 가짐

**PixelCNN**
<img src="https://user-images.githubusercontent.com/59776953/119319563-9338ae80-bcb5-11eb-9a95-94d48c2e806d.jpg" width="50%">

- 코너에서 시작하여 pixel을 생성한다는 점에서 PixelRNN과 공통점을 가짐
- context 영역에 CNN을 적용하여 pixel을 생성할 때 특정 pixel만을 고려
	- 이전의 pixel에 대한 의존성을 바탕으로 함
- 이러한 과정을 통해 likelihood를 최대화
- PixelRNN보다 빠르다
	- Training 이미지의 context 영역 값으로 convolution을 병렬화한다
	- 순차적으로 생성하기 때문에 여전히 느림

**Generations Sample**
<img src="https://www.dropbox.com/s/6js5vkewwvc1s68/Screenshot%202018-06-10%2011.04.33.png?raw=1">

## Variational Autoencoders (VAE)

<img src="https://www.dropbox.com/s/cg0673i8cayx62g/Screenshot%202018-06-10%2011.14.03.png?raw=1">

- 굉장히 복잡하기 때문에 직접적으로 최적화할 수 없고, likelihood에 대한 하한으로 유도하고 최적화

**Encoder and Decoder Network in VAE**

<img src="https://www.dropbox.com/s/npg2xnhb81t645u/Screenshot%202018-06-10%2011.25.52.png?raw=1">

- encoder는 지도학습 모델을 initialize하기 위해 사용 가능
- autoencoder를 통해 새로운 이미지 생성 가능 -> VAE

## Generative Adversarial Networks(GAN)

- 이전에는 확률분포를 explicit하게 모델링한 것과 달리 이를 포기하고 샘플을 만드는데 중점적으로 둔 모델(implicit modeling)
- 게임이론의 접근방식을 취하여 2-player game 방식으로 training distribution을 학습

## Training GANs: Two-player game

<img src="https://www.dropbox.com/s/2fkftsb2sksw8ue/Screenshot%202018-06-11%2021.33.15.png?raw=1">

**기본 원리**
- generator network : 실제와 비슷한 이미지를 생성하여 discriminator를 속임
- discriminator network :  실제와 가짜 이미지를 구별해 냄

**학습 과정**
<img src="https://younghk.netlify.app/static/8ec32de80a25aeb363918b39d996a0bf/0cbdd/image47.png">
- discriminator를 먼저 학습: 진짜 이미지가 들어가면 진짜로 구분, 가짜 이미지가 들어가면 가짜로 구분
	- input: 이미지의 고정된 벡터
	- output: 진짜 / 가짜 (sigmoid를 통하여 0.5 기준으로 classification)
- generator가 이미지를 생성하여 discriminator가 1이 나오도록 학습

**상반된 목적성**
<img src="https://www.dropbox.com/s/3wkocm1qvvxkl0i/Screenshot%202018-06-11%2021.39.28.png?raw=1">
 - discriminator는 목적함수가 최대화하는 것을 목표
	 - D(x)는 1에 가깝고 D(G(z))의 경우 0에 가까워야 함
 - generator는 목적함수를 최소화하는 것을 목표
	- D(G(z))가 1에 가까워야 함
	- gradient ascent를 이용하여 D(G(z))의 평균을 최대화
	 
 -> 게임 이론

**Generated Sample**
<img src="https://www.dropbox.com/s/3fwxj66svo68s39/Screenshot%202018-06-11%2022.34.12.png?raw=1">
<img src="https://www.dropbox.com/s/c2aeog6hwmlo9d4/Screenshot%202018-06-11%2022.34.28.png?raw=1">

## GAN 응용

<img src="https://younghk.netlify.app/static/68910bc2fff198b3aa1d59df2990c5f1/36c39/image52.png">

Original GAN(2014) 이후 GAN의 성능을 향상시키디 위하여 A.Radford가 ICLR'16에 발표한 CNN 아키텍쳐를 GAN에 적용한 연구로 GAN의 성능을 극적으로 끌어올렸다.

DCGAN에서 GAN에 적용한 CNN 아키텍쳐를 살펴보면 입력 노이즈 벡터 z가 있고 z를 다음과 같은 과정으로 sample 출력으로 변환한다.

<img src="https://younghk.netlify.app/static/051cddf57bf3fae36588417cc9b9ad34/79d86/image53.png">

위의 사례에서는 z 포인트를 두개로 잡아 그 사이를 interpolate를 하여 이미지를 생성하였다. 두 이미지 간에 이미지가 부드럽게 변하는 것을 확인할 수 있다.

<img src="https://younghk.netlify.app/static/f19e0956b5c473f46e7415cd12dd476b/4d248/image55.png">

벡터의 연산을 이용하여 다음과 같은 예시를 만들어 낼 수 있다.
