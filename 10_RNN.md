
# Recurrent Neural Networks

## RNN 개요

**주식**

![주식 그래프](https://file.mk.co.kr/meet/neds/2021/01/image_readtop_2021_30235_16103273954502806.jpg)

**영상**

![영상](https://t1.daumcdn.net/cfile/tistory/99DEE74C5DC69E9001)

- 음악, 동영상, 주가차트 등과 같은 순서대로 처리해야 하는 시퀀스 데이터를 모델링 하기 위해 등장
- 기억(hidden state)를 갖고 있다는 점에서 기존의 신경망과 차별점을 가짐

**RNN**

![RNN](http://i.imgur.com/s8nYcww.png)

- input x: t시점에서의 input
- hidden state h: 현재 상태인 t시점의 h(t)는 직전 시점의 히든 state h(t−1)를 받아 갱신
- output y: h(t)를 전달받아 가중치를 곱하여 산출

## RNN applications

**Process Sequences**

![process sequences](https://i.stack.imgur.com/b4sus.jpg)

- one to one: RNN 모델의 가장 단순한 형태 
	- input -> hidden -> output

- one to many: Image Captioning에 주로 사용되는 형태
- 
![image captioning](https://github.com/danieljl/keras-image-captioning/raw/master/results-without-errors.jpg)

	- image -> sequence of words

- many to one: Sentiment Classification 감정 분류에 주로 사용되는 형태
- 
![sentiment classification](https://wikidocs.net/images/page/44249/navermovie5.PNG)

	- sequece of words -> positive/negative

- many to many 1: Translation에 주로 사용되는 형태
	- sequence of words -> sequence of words
	- 한국 단어 문장 -> 영어 단어 문장

- many to many 2: Video Classification on Frame Level 에 주로 사용되는 형태
	- frame이 들어옴에 따라 바로바로 분류
	- 각각 time step에서의 예측은 현재의 frame + 지나간 frame들에 대한 함수

## RNN 기본 동작

**character-level-model**

![character-level-model](http://i.imgur.com/vrD0VO1.png)

**RNN front propagation**

![RNN front propagation](http://i.imgur.com/TIdBDTJ.png)

**RNN back propagation**

![RNN back propagation](http://i.imgur.com/Xtpgxzu.gif)

one-hot encoder을 통하여 'h', 'e', 'l', 'o'를 각각 [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]로 치환한다. front propagation을 거치고 output이 각각 초록색 target character가 도출될 수 있도록 back propagation을 수행하여 parameter 값들을 갱신한다. 

모든 시점의 state에서 parameter는 동일하게 적용된다.(shared weights)

## LSTM

RNN의 경우에 기억하는 범위가 너무 많아지게 되면 back propagation시 gradient가 점차 줄어 학습 능력이 저하된다. 이를 vanishing gradient problem이라고 한다. 또한 Gradient 계산이 너무 복잡해져 계산량이 폭발적으로 증가하게 되어서 기억할 범위를 조금 전으로 한정하고 그 이전의 것을 버리고 있다. 이를 Truncated Back Propagation Through Time이라고 한다.

하지만 더 이전의 정보를 이용하지 않으면 작동하지 않는 경우가 많은데 이를 해결하기 위하여 등장한 것이 Long Short-Term Memory(LSTM)이라는 기술이다. 

**RNN vs LSTM**

![RNN](http://i.imgur.com/jKodJ1u.png)

LSTM은 RNN의 hidden state에 cell state를 추가한 구조이다. cell state가 일종의 컨베이어 벨트 역할을 하여 state가 오래 경과하더라도 gradient가 잘 전파되도록 한다.

![lstm formula](https://user-images.githubusercontent.com/59776953/117636547-e3931500-b1bb-11eb-953d-067b4d6114fc.png)
![lstma formula2](https://user-images.githubusercontent.com/59776953/117637507-d165a680-b1bc-11eb-8864-b52b4dc9cc4c.png)

forget gate f를 통하여 '과거 정보를 얼마나 잊을 것인지'를 지정해준다. 이전 시점의 hidden layer h(t-1)과 x(t)를 받아서 시그모이드를 취하여 내보내는 값으로 0이면 이전 상태의 정보를 잊고, 1이면 이전 상태의 정보를 온전히 기억하게 된다.

c(t) 계산에서 forget의 연산 이후에 이어지는 것은 '현재 정보를 기억하기 위한 게이트'라고 생각하면 된다. i(t)의 범위는 0에서 1까지고 g(t)의 범위는 -1에서 1사이를 나타낸다.

cell state를 구하여 하이퍼볼릭 탄젠트를 취하고 o(t)값과 연산하여 hidden state로 전달하게 된다. o(t)의 경우 현재의 cell state 중에 어느 부분을 다음의 hidden state로 전달할지를 판단해주는 것이라고 생각하면 된다. hidden vector는 LSTM의 다음 iteration과 상위 layer or prediction으로 이동하게 된다.

### 출처

[Long Short-Term Memory (LSTM) 이해하기 :: 개발새발로그 (tistory.com)](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr)

[LSTM을 활용한 주식가격 예측 - Data Science | DSChloe](https://dschloe.github.io/python/python_edu/07_deeplearning/deep_learning_lstm/#lstm%EA%B3%BC-rnn%EC%9D%98-%EA%B0%9C%EC%9A%94)

[RNN과 LSTM을 이해해보자! · ratsgo's blog](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/)

[Long Short-term Memory | Sepp Hochreiter](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
