# stacked_hourglass
a pytorch implementation of 'Stacked Hourglass Networks for Human Pose Estimation'


## stacked_hourglass 요약

이미지의 모든 scale에 대한 정보를 downsampling 과정에서 추출하고 이를 upsampling 과정에 반영하여 pixel-wise output을 생성하는 것을 목표로 한다

Hourglass module을 8개 연속하여 잇는 Stacked Hourglass Network 구조를 가진다

## stacked_hourglass 구조 설명

최종적으로 pose를 추정하기 위해서는 full body에 대한 이해가 매우 중요한데, 이를 위해서 여러 scale에 걸쳐 필요한 정보를 포착해낼 수 있어야한다.

Hourglass는 이런 모든 feature를 잡아내어 Network의 출력인 pixel 단위의 예측에 반영되도록 한다.
그리고 Skip layer를 이용하여 한개의 pipline 만으로 spatial information을 유지하는 방식을 채택한다.

feature 추출과 저차원으로의 downsampling(이미지 작게) 을 위해 Convolutional & Pool layer 사용

가장 낮은 resolution 도달 후, unsampling(이미지 크게) 과정에서 scale 별로 추출한 feature 조합.

대칭적인 구조를 가진다.

Output resolution에 다다르면, 2번 연속 1x1 convolution 연산 적용 후 최종 출력을 한다.

