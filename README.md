# Stacked_Hourglass
About Stacked Hourglass Network

## 01. Stacked Hourglass 요약  

이미지의 모든 scale에 대한 정보를 downsampling 과정에서 추출하고 이를 upsampling 과정에 반영하여 pixel-wise output을 생성하는 것을 목표로 한다

Hourglass module을 8개 연속하여 잇는 Stacked Hourglass Network 구조를 가진다

## 02. Stacked Hourglass 구조 설명 

최종적으로 pose를 추정하기 위해서는 full body에 대한 이해가 매우 중요한데, 이를 위해서 여러 scale에 걸쳐 필요한 정보를 포착해낼 수 있어야한다.

Hourglass는 이런 모든 feature를 잡아내어 Network의 출력인 pixel 단위의 예측에 반영되도록 한다.
그리고 Skip layer를 이용하여 한개의 pipline 만으로 spatial information을 유지하는 방식을 채택한다.

feature 추출과 저차원으로의 downsampling(이미지 작게) 을 위해 Convolutional & Pool layer 사용

가장 낮은 resolution 도달 후, unsampling(이미지 크게) 과정에서 scale 별로 추출한 feature 조합.

대칭적인 구조를 가진다.

Output resolution에 다다르면, 2번 연속 1x1 convolution 연산 적용 후 최종 출력을 한다.

## 03. 1개의 Hourglass

![stackedhourglass1 PNG](https://user-images.githubusercontent.com/80568500/152296722-48a8ef76-d05c-4372-98bd-1a348d4df3a4.jpg)

## 04. 2개의 Stacked Hourglass

![2stacked](https://user-images.githubusercontent.com/80568500/152296751-7f7ee09b-d82e-4a50-a958-d5bfda5e0f94.jpg)

다수의 Hourglass를 쌓아놓은 구조가 Stacked Hourglass Network이다. 이를 통해 initial estimate과 이미지 전반에 대한 feature를 다시금 추정할 수 있게 한다.

중간 얻어지는 예측값(heatmaps)에 대해서도 ground truth 와의 loss를 적용하여 intermediate supervision을 가능하게 한다.

반복적인 예측값의 조정으로 좀 더 세밀한 결과를 도출할 수 있으며, 중간 중간 적용되는 loss 로 인해 좀 더 깊고 안정적인 학습이 가능하다.

※ Hourglass Module간 weight 공유 하지 않는다!
