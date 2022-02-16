# TENSORFLOW CUDA SETTING FOR WINDOWS


##### 방법 1. 그래픽 드라이버 , CUDA Tool kit, cuDNN 설치 후 환경 변수 설정
* 이 방법은 비추 
* 텐서플로 모듈 관련 에러 자주발생, 버전 맞추기 번거로움
---------------------------

#### 방법 2. (추천) Anaconda 설치 후 conda 가상환경에서 tf 인스톨
* 아나콘다 가상환경에서 tensorflow-gpu 인스톨
* CUDA tool kit, cuDNN 인스톨 목록에 있는 지 확인
* 자동으로 버전 맞춰지며 gpu 사용 가능


CPU : RYZEN 5 5600x (4.6GHZ)

GPU : Geforce RTX 3060

"""

*샘플 코드 실행 결과*

### GPU 

Accuracy:  1.0

훈련 끝

소요시간 : 0시 1분 37초

----------------------
### CPU

Accuracy:  1.0

훈련 끝

소요시간 : 0시 4분 23초

"""