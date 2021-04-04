# 2. Image Classification

## Computer Vision and Image Classification

**컴퓨터 비전(Computer Vision)**
 - 기계의 시각에 해당하는 부분을 연구하는 분야
 - 공학적인 관점에서, 인간의 시각이 할 수 있는 몇가지 일을 수행하는 자율적인 시스템을 만드는 것을 목표

 분야: 이미지 분류, 위치 인식, 물체 검출, 이미지 캡셔닝

**이미지 분류(Image Classification)**
![Image Classification](https://user-images.githubusercontent.com/59776953/113523115-de511200-95e0-11eb-98cd-cef16b96d2de.png)
이미지 전체 혹은 이미지 안의 물체(object)의 종류를 구분하는 작업

**물체 위치인식(Object Localization)**
![localization](https://user-images.githubusercontent.com/59776953/113523118-e27d2f80-95e0-11eb-856a-53cffb9b7c49.png)

이미지 안의 물체가 이미지의 어느 영역에 있는지 위치 정보를 출력해주는 작업

**물체 검출(Object Detection)**
![Object Detection](https://user-images.githubusercontent.com/59776953/113523127-eb6e0100-95e0-11eb-8f33-693db307764f.png)
물체가 무엇인지 분류(classification)하는 과정과 물체가 어디에 있는지 위치 정보를 알려주는 위치인식(localization) 과정이 동시에 수행되는 작업

**이미지 캡셔닝(Image Captioning)**
 ![Image Captioning](https://user-images.githubusercontent.com/59776953/113523113-dd1fe500-95e0-11eb-8461-2120cea7e0d1.png)
 이미지를 설명하는 문장을 만들어내는 작업
 
## Challenges of Image Classification

![problem](https://user-images.githubusercontent.com/59776953/113523170-28d28e80-95e1-11eb-9554-d43c281698ae.png)
**"숫자로 구성된 3D Array(height x width x color channel)를 기반으로 어떻게 이미지 분류를 할까?"**

Challenges
1. Viewpoint Variation - 시각에 따라 다르게 보인다
2. Illumination - 사진과 배경의 밝기
3. Deformation - 형태의 변형
4. Occulusion - 가려져 있거나 일부만 보일 때
5. Background Clutter - 배경과 구분이 잘 안될 때
6. Interclass Variation - 다양한 종을 모두 판단해야 함

## Rule-Based Approach vs. Data-Driven Approach

**Rule-Based Approach**

초기에는, 특정한 edge, shape, junction을 찾아 특정 알고리즘을 통하여 판단하는 Rule-Based Approach를 사용하였다. 

문제점
1. 규칙을 만들어내기 어려움
2. 확장성(Scalability)가 떨어짐

![rule_vs_data](https://user-images.githubusercontent.com/59776953/113523350-579d3480-95e2-11eb-9a4c-94e16923d7f1.png)

**Data-Driven Approach**

사람이 직접 알고리즘을 만드는게 아니라, 데이터를 기반으로 모델을 만들어 문제를 해결하는 방법

1. 라벨화된 이미지를 수집한다
2. 머신러닝을 이용하여 Classifier을 train한다
3. 새로운 이미지를 기반으로 Classifier의 성능을 평가한다

## Nearest Neighbor

**Nearest Neighbor Search**
![Nearest Neighbor](https://user-images.githubusercontent.com/59776953/113523304-042ae680-95e2-11eb-9f76-c3823b7d0dc6.png)
가장 가까운 점을 찾기 위한 최적화 문제

거리 계산 방식
**L1 Distance(Manhattan Distance)**
![l1 distance](https://user-images.githubusercontent.com/59776953/113523229-7e0ea000-95e1-11eb-9196-7e00b47d63ce.png)
두 개의 벡터를 빼고 절대값을 취한 뒤 합하는 방식

![nearest neighbor code](https://user-images.githubusercontent.com/59776953/113523216-6b946680-95e1-11eb-94ff-eeb6b0d762bd.png)
```python
import numpy as np class
NearestNeighbor(object):  
	def __init__(self):  
		pass  
	def train(self, X, y):  
		""" X is N x D where each row is an example.Y is 1-dimension of size N """  
		# the nearest neighbor classifier simply remembers all the training data 
		self.Xtr = X 
		self.ytr = y 
	def predict(self, X):  
		""" X is N x D where each row is an example we wish to predict label for """ 
		num_test = X.shape[0]  
		# lets make sure that the output type matches the input type 
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)  

		# loop over all test rows  
		for i in  xrange(num_test):  
			# find the nearest training image to the i'th test image  
			# using the L1 distance (sum of absolute value differences) 
			distances = np.sum(np.abs(self.Xtr - X[i,:]), axis =  1) 
			min_index = np.argmin(distances)  # get the index with smallest distance 
			Ypred[i]  = self.ytr[min_index]  # predict the label of the nearest example  

		return Ypred  
```

문제점
- Nearest Neighbor의 경우, 메모리 상에 training data와 label을 모두 올리고 test 데이터와의 distance를 살핌
- training data가 2배 늘어나면 분류 작업 시간도 2배 늘어나게 됨 🠖 test 시, 오래 걸림
- CNN에서는 training에는 오래 걸리지만, test 시 굉장히 짧게 걸림

## k-Nearest Neighbors

**k-NN Classification**
 특징 공간 내 k개의 가장 가까운 훈련 데이터 사이에서 가장 공통적인 항목에 할당

**Hyperparameter**

거리 계산 방식
![manhattan_euclidean](https://user-images.githubusercontent.com/59776953/113523121-e4df8980-95e0-11eb-988c-fddcb372ee76.png)
**L1 Distance(Manhattan Distance)**
두 개의 벡터를 빼고 절대값을 취한 뒤 합하는 방식

**L2 Distance(Euclidean Distance)**
두 개 벡터 사이의 직선 거리

k값 
![k](https://user-images.githubusercontent.com/59776953/113523427-86b3a600-95e2-11eb-8711-2b6b64b96ac7.png)

위의 그림에서 kNN이 NN보다 이상치에 둔감한 것을 확인할 수 있다. k가 높아지면 그만큼 계산량이 많아지는 단점이 있다.

- 적절한 k값 정하기
- L1 Distance/L2 Distance 방식 중 더 나은 방식 선택하기

## Hyperparameter 설정

**Hyperparameter 설정을 위해 Validation Data 마련**

**교차 검증(Cross Validation)**
![k-fold cross validation](https://user-images.githubusercontent.com/59776953/113523090-bcf02600-95e0-11eb-8cba-e96d1b4a2716.png)

고정된 training set으로 모델을 만들 경우 과적합이 일어날 수 있는데 이를 방지하기 위해 교차 검증을 사용한다

K-Fold Cross Validation 과정
1. 전체 데이터셋을 training set과 test set으로 나눈다
2. training set을 training set + validation set으로 사용하기 위해 k개의 fold로 나눈다
3.  첫 번째 fold를 validation set으로 사용하고 나머지 fold들을 training set으로 사용한다
4. 모델을 training한 뒤, 첫 번째 validation set으로 평가한다
5. 차례대로 다음 fold를 validation set으로 사용하며 3을 반복한다
6. 총 k 개의 성능 결과가 나오며, 이 k 개의 평균을 해당 학습 모델의 성능이라고 한다

## Linear Classification

 **선형 분류(Linear Classification)**
 선을 이용하여 입력값에 대해 여러 class 중 하나를 택해 분류하는 모델
 - parameter 기반의 접근 방식
 (NN은 nonparametric approach)

![linear classifier](https://user-images.githubusercontent.com/59776953/113523266-c1690e80-95e1-11eb-8ef2-5111cf434726.png)

![linear classifier2](https://user-images.githubusercontent.com/59776953/113523267-c332d200-95e1-11eb-9c79-c925a4c7886d.png)

Classifier가 내놓은 결과 값에 대해 제대로 분류했나 평가하기 위하여 정답 label과 비교를 하는데 이 때 비교하는 함수가 **loss function**

- Y = W x + bias 좌표 상 한 직선으로 나누지 못하면 linear classifier가 통하지 않음

출처
- https://m.blog.naver.com/arar2017/221791751470
- https://3months.tistory.com/512
- https://ko.wikipedia.org/wiki/K-%EC%B5%9C%EA%B7%BC%EC%A0%91_%EC%9D%B4%EC%9B%83_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
- https://ko.wikipedia.org/wiki/%EC%B5%9C%EA%B7%BC%EC%A0%91_%EC%9D%B4%EC%9B%83_%ED%83%90%EC%83%89
