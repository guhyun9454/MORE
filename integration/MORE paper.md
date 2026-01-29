제공해주신 논문 "A Multi-Head Model for Continual Learning via Out-of-Distribution Replay"에 기반하여 MORE (Multi-head OOD REplay) 알고리즘의 구현을 위한 상세한 설명을 드리겠습니다.
이 방식은 기존의 Replay 방식(과거 데이터를 학습에 재사용하여 기억을 유지하는 방식)과 달리, 과거 데이터를 '현재 태스크의 분포가 아닌(OOD, Out-of-Distribution) 데이터'로 간주하여 현재 태스크 모델이 이를 거부하도록 학습시키는 것이 핵심입니다.

구현은 크게 모델 아키텍처, 학습 단계(Training), 백-업데이트(Back-updating), **추론 단계(Prediction)**의 네 부분으로 나뉩니다.
1. 모델 아키텍처 (Model Architecture)
MORE는 사전 학습된(Pre-trained) 트랜스포머 모델을 기반으로 하며, 다음과 같은 구조적 특징을 가집니다.
백본(Backbone): 사전 학습된 Vision Transformer(ViT)를 사용합니다. 이때 트랜스포머의 원래 파라미터 $\theta$는 고정(frozen)되어 변경되지 않습니다.


어댑터(Adapter): 각 트랜스포머 레이어에 학습 가능한 '어댑터 모듈'을 삽입합니다. 이 어댑터 파라미터와 Layer Norm 파라미터만 학습됩니다.


어댑터 구조: [FC Layer (Projection to bottleneck) -> ReLU -> FC Layer (Projection back)] 구조를 가집니다.


멀티 헤드 분류기(Multi-head Classifiers): 각 태스크 $k$마다 별도의 분류기 헤드(Classifier Head) $\phi_k$를 가집니다.


각 헤드의 출력 클래스 개수는 $|\mathcal{Y}^k| + 1$개입니다. 여기서 마지막 $+1$ 클래스는 OOD 클래스(과거 또는 미래의 보지 못한 클래스)를 의미합니다.


2. 학습 단계 (Training Phase)
태스크 $k$가 주어졌을 때, 현재 태스크의 데이터 $\mathcal{D}^k$와 메모리 버퍼에 저장된 과거 데이터 $\mathcal{M}$을 사용하여 학습을 진행합니다.
2.1 데이터 구성 및 레이블링
IND (In-Distribution) 데이터: 현재 태스크 $\mathcal{D}^k$의 샘플입니다. 이들은 해당 클래스의 정답 레이블 $y$를 갖습니다.
OOD (Out-of-Distribution) 데이터: 메모리 버퍼 $\mathcal{M}$에 저장된 과거 태스크의 샘플들입니다. 이들은 현재 태스크 관점에서는 OOD이므로, 모두 OOD 레이블(추가된 마지막 클래스 인덱스)을 갖습니다.


2.2 손실 함수 (Loss Function)
태스크 $k$에 대한 분류기 $\phi_k$와 어댑터 파라미터를 학습하기 위해 다음 손실 함수를 최소화합니다.

$$\mathcal{L}_{ood}(\theta, \phi_k) = - \frac{1}{M+N} \left( \sum_{(x,y) \in \mathcal{M}} \log p(ood|x,k) + \sum_{(x,y) \in \mathcal{D}^k} \log p(y|x,k) \right)$$


즉, 현재 태스크 데이터는 올바른 클래스로 분류하고, 메모리 버퍼 데이터는 'OOD 클래스'로 분류하도록 학습합니다.
2.3 Hard Attention (HAT) 마스킹
이전 태스크의 지식을 잊지 않기 위해(Catastrophic Forgetting 방지), HAT(Hard Attention to the task) 메커니즘을 어댑터에 적용합니다.
마스크 생성:
각 레이어 $l$과 태스크 $k$에 대해 학습 가능한 임베딩 $e^k_l$을 정의하고, 시그모이드 함수를 통해 마스크 $a^k_l$을 생성합니다 ($s$는 큰 양의 상수, 예: 500).

$$a^k_l = \sigma(s \cdot e^k_l)$$


마스크 적용:
이 마스크는 해당 레이어의 출력 $h_l$과 요소별 곱(element-wise multiplication)을 수행하여 정보 흐름을 제어합니다.

$$h'_l = a^k_l \otimes h_l$$


그라디언트 수정 (Gradient Modification):
이전 태스크들($1, ..., k-1$)에서 중요하다고 판단된 파라미터가 변경되지 않도록 그라디언트를 수정해야 합니다. 누적 마스크 $a^{<k}_l$ (이전 태스크들의 마스크 최댓값)을 사용하여 업데이트를 차단합니다.

$$\nabla w'_{ij,l} = (1 - \min(a^{<k}_{i,l}, a^{<k}_{j,l-1})) \nabla w_{ij,l}$$


$a^{<k}$ 값이 1에 가까우면(이전에 사용된 뉴런이면), $(1-1)=0$이 되어 그라디언트가 0이 되고 파라미터가 업데이트되지 않습니다.
Sparsity Regularization:
네트워크 용량을 효율적으로 사용하기 위해 마스크의 희소성(sparsity)을 유도하는 규제항 $\mathcal{L}_r$을 추가합니다.

$$\mathcal{L}_r = \lambda \frac{\sum_l \sum_i a^k_{i,l}(1 - a^{<k}_{i,l})}{\sum_l \sum_i (1 - a^{<k}_{i,l})}$$


최종 목적 함수:

$$\mathcal{L} = \mathcal{L}_{ood} + \mathcal{L}_r$$


3. 백-업데이트 단계 (Back-Updating Phase)
현재 태스크 $k$의 학습이 끝난 직후, **이전 태스크들의 분류기($\phi_1, ..., \phi_{k-1}$)**를 업데이트하여 현재 태스크 데이터를 OOD로 잘 인식하도록 만듭니다. 이는 초기 태스크 모델들이 다양한 OOD 데이터를 보지 못해 과신(over-confidence)하는 문제를 해결합니다.

절차:
현재 태스크 데이터 $\mathcal{D}^k$에서 메모리 버퍼 크기($M$)만큼의 샘플을 무작위로 추출하여 $\tilde{\mathcal{D}}$를 만듭니다.


이전 태스크 $j (<k)$에 대해, $\tilde{\mathcal{D}}$와 메모리 버퍼 $\mathcal{M}$ (태스크 $j$의 IND 데이터 제외)을 합쳐 새로운 OOD 데이터셋 $\mathcal{M}'$을 구성합니다.


태스크 $j$의 분류기 헤드 $\phi_j$만 업데이트하고(Feature Extractor 고정), 다음 손실 함수를 최소화합니다.

$$\mathcal{L}(\phi_j) = - \frac{1}{2M} \left( \sum_{(x,y) \in \mathcal{M}'} \log p(ood|x,j) + \sum_{(x,y) \in \bar{\mathcal{D}}^j} \log p(y|x,j) \right)$$


$\bar{\mathcal{D}}^j$: 태스크 $j$의 정답 데이터 (메모리에 저장된 것).
4. 추론 단계 (Inference/Prediction Phase)
테스트 시에는 입력 데이터가 어떤 태스크에 속하는지 알 수 없으므로(Task-ID 없음), 모든 태스크 헤드의 출력을 비교해야 합니다. 이때 **마할라노비스 거리(Mahalanobis Distance)**를 기반으로 한 보정 계수를 사용합니다.
4.1 통계량 계산 (학습 종료 시 저장)
각 태스크 학습이 끝날 때, 각 클래스 $j$의 평균($\mu^k_j$)과 태스크 $k$의 공분산($S^k$)을 계산하여 저장해 둡니다.

$$\mu^k_j = \frac{1}{|\mathcal{D}^k_j|} \sum_{x \in \mathcal{D}^k_j} h(x, k)$$

$$S^k = \frac{1}{|\mathcal{Y}^k|} \sum_{j \in \mathcal{Y}^k} S^k_j$$


4.2 거리 기반 계수 계산
테스트 샘플 $x$에 대해, 태스크 $k$에 속할 가능성을 나타내는 계수 $s^k(x)$를 계산합니다. 이는 해당 태스크 내 모든 클래스와의 마할라노비스 거리 중 최솟값의 역수를 취한 것입니다. ($c$는 상수, 예: 20)

$$s^k(x) = \max \left[ \frac{c}{MD(x; \mu^k_{y_1}, S^k)}, \dots, \frac{c}{MD(x; \mu^k_{y_{|\mathcal{Y}^k|}}, S^k)} \right]$$


4.3 최종 예측
각 태스크 $k$의 소프트맥스 확률 $p(\mathcal{Y}^k|x,k)$ (OOD 클래스 제외)에 위에서 구한 계수 $s^k(x)$를 곱합니다. 모든 태스크의 결과값을 연결(concatenate)한 후, 가장 큰 값을 가진 클래스를 최종 예측으로 선택합니다.

$$y = \arg\max \bigoplus_{1 \le k \le t} p(\mathcal{Y}^k|x,k) \cdot s^k(x)$$


요약: 구현 체크리스트
Backbone: ImageNet Pre-trained ViT 준비 (Frozen).
Network: 각 레이어에 Adapter와 Task-specific Mask(Embedding) 추가.
Heads: 태스크마다 별도의 FC Layer (Output dim = Class 수 + 1).
Forward Pass: 입력 -> Adapter & Mask 적용 -> Head -> Logits.
Loss: CrossEntropy (IND 데이터는 정답 클래스, Memory 데이터는 Last 클래스).
Optimization: $\mathcal{L}_{ood} + \mathcal{L}_r$로 Adapter와 Head 업데이트 (Masking된 Gradient 적용).
Post-Task: 현재 데이터를 OOD로 삼아 이전 태스크 헤드들만 Fine-tuning (Back-update).
Prediction: (Softmax Probability $\times$ Inverse Mahalanobis Distance)가 최대인 클래스 선택.

1. 일반 설정 (Common Settings)모든 실험에 공통적으로 적용된 기본 설정입니다.백본 네트워크 (Backbone): DeiT-S/16 (Vision Transformer) 어댑터 구조 (Adapter): 각 트랜스포머 레이어마다 2-layer 어댑터 사용 (Houlsby et al., 2019 구조) 최적화 함수 (Optimizer): SGD (Stochastic Gradient Descent) 모멘텀 (Momentum): 0.9 거리 기반 계수 상수 ($c$): 20 (Eq. 4에서 사용) Hard Attention 관련:Sigmoid 스케일링 파라미터 ($s$): 500 (Eq. 8에서 사용) Sparsity 규제 가중치 ($\lambda$): 0.75 (Eq. 12에서 사용) 2. 데이터셋별 설정 (Dataset Specifics)데이터셋 및 태스크 분할(Task Split)에 따라 하이퍼파라미터가 다르게 설정되었습니다.항목CIFAR-10 (5 Tasks)CIFAR-100 (10 Tasks)CIFAR-100 (20 Tasks)Tiny-ImageNet (5 Tasks)Tiny-ImageNet (10 Tasks)Epochs20 40 40 15 10 Learning Rate0.005 0.001 0.005 0.005 0.005 Memory Size200 (Total) 2,000 (Total) 2,000 (Total) 2,000 (Total) 2,000 (Total) Adapter Bottleneck64 128 (64의 2배) 128 128 128 Back-Update 적용 여부적용함 적용함 적용함 적용 안 함 적용 안 함 3. 백-업데이트 설정 (Back-Updating Settings)이전 태스크 모델을 업데이트할 때(Section 3.2) 사용하는 별도의 하이퍼파라미터입니다. 이 설정은 CIFAR-10과 CIFAR-100 실험에만 적용되며, Tiny-ImageNet에는 적용되지 않습니다.Epochs: 10 Learning Rate: 0.01 Batch Size: 16 Momentum: 0.9 4. 메모리 관리메모리 버퍼: 각 클래스당 동일한 수의 랜덤 샘플을 저장합니다.새로운 태스크가 추가될 때마다 버퍼 크기를 고정하고 저장된 샘플 수를 줄여서 새로운 샘플을 수용합니다.