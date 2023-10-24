# 딥러닝 모델을 활용한 중고상품 분류모델 개발
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/Visual Studio Code-007ACC?style=flat-square&logo=Visual Studio Code&logoColor=white"/> <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white"/> <img src="https://img.shields.io/badge/JSON-000000?style=flat-square&logo=json&logoColor=white"/> <img src="https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=MySQL&logoColor=white"/> 
<img src="https://img.shields.io/badge/scikit-learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/> <img src="https://img.shields.io/badge/streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>

(제 개인레포에 파이널 프로젝트 파일 등을 다시 정리했습니다)

### 프로젝트 기간: 6/1 ~ 6/30

### 팀소개
► 함께한 동료들: 이나라 (팀장), 류영주, 임희나, 장하은, 최원빈, 하정현

► [<img src="https://img.shields.io/badge/Notion-000000?style=flat-square&logo=Notion&logoColor=white">](https://www.notion.so/Final-Project-c05e51833a11418eb27efbc197c9f1da?pvs=4) (팀 노션 링크)

► [발표자료 링크](https://drive.google.com/file/d/1X2TrWvrmMYey0RCPzNO6URM-5iMC_BjP/view?usp=sharing)

### 프로젝트 우수상 수상

<img src="https://github.com/owenchoi96/templates/assets/123911225/32a35169-44df-4561-9275-028b7f6f2c30" width="200" height="300"/>

## 프로젝트 선정 계기
<p align="center">
<img src="https://github.com/owenchoi96/templates/assets/123911225/573d8895-618b-4927-9958-e0935c638685" width="400" height="250"/>
</p>

최근 뉴스를 통해 중고거래 시장의 지속적인 성장을 발견했습니다. 그래서 성장하는 시장에서 분석 및 개발을 통해 해결할 수 있는 문제가 무엇일지 고민하던 중 중고거래 상품의 분류에 니즈가 있는 것을 발견했습니다. 중고거래가 증가함에 따라 업로드되는 상품 수가 증가할 것으로 예상이 되었습니다. 그런데 상품을 유저들이 직접 올리고 카테고리도 직접 설정하기 때문에 카테고리 분류의 오분류가 일어날 수 있음을 발견했습니다. 카테고리 분류가 잘되지 않는다면 상품 추천 알고리즘 성능 저하 및 유저가 상품을 찾는데 어려움을 겪을 수 있기 때문에 문제해결의 필요성을 느꼈습니다. 

## 진행과정
<p align="center">
<img src="https://github.com/owenchoi96/templates/assets/123911225/5de532d3-7df3-4136-9420-662db6841a98" width="400" height="250"/>
</p>

1. 먼저 requests 라이브러리를 활용하여 번개장터와 네이버 쇼핑의 상품명과 이미지 데이터를 크롤링한 후 수집한 데이터를 DB에 저장했습니다.
2. 수집한 데이터의 중복을 제거하고 모델링에 활용하기 위한 클리닝 작업을 수행했습니다.
3. 머신러닝, 딥러닝 모델을 돌리기 위해 데이터를 레이블링하는 과정을 거쳤습니다
4. 마지막으로 텍스트 딥러닝 모델과 이미지 딥러닝 모델을 각각 그리고 합친 앙상블 모델을 활용하여 모델링을 진행했습니다. 

<p align="center">
<img src="https://github.com/owenchoi96/templates/assets/123911225/ecbb135a-6365-4c93-a815-3ca28eb7bbc8" width="400" height="250"/>
</p>

모델링 결과는 위와 같습니다. 텍스트 분류를 위한 다른 방식으로 작동하는 3가지 타입의 언어모델들을 활용하여 모델링을 진행했습니다. 157개 카테고리를 중 한가지의 패션 카테고리를 예측하는 모델링을 했으시 모든 분류 모델의 성능이 60% 아래로 떨어지는 것을 확인했습니다. 때문에 한개의 카테고리가 아닌 3개의 카테고리를 예측하는 모델로 방향을 바꾸어 모델링을 진행했습니다. 그 결과 KoElectra_base 모델과 KoBert_Transformers가 높은 성능을 내는 것을 발견했습니다. 

이미지 모델의 경우 모델링에 시간이 너무 오래 걸릴 것을 우려하여 가벼운 모델인 MobileNetV2를 사용하여 모델링을 진행했습니다. 이미지 모델 역시 하나의 카테고리를 예측하는 모델보다 3개의 카테고리를 예측하는 모델에서 더 좋은 성능을 나타내는 것을 발견했습니다. 

그래서 두가지 모델을 합쳐 앙상블 모델을 만드려고 했으나 모델의 성능은 텍스트 모델 성능보다 더 좋아지지는 않는다는 것을 발견했습니다 (약 84%의 성능). 

## 개인적으로 아쉬웠던 부분 2가지
사실 프로젝틀 끝나고 인사이트에 관한 내용을 적고 싶었습니다. 하지만 처음으로 진행했던 데이터 분석 및 개발 프로젝트인만큼 부족한 점이 많았습니다. 그래서 아쉬웠던 부분 2가지를 다루어보고자 합니다. 

<p align="center">
<img src="https://github.com/owenchoi96/templates/assets/123911225/42afc0dc-00a7-4734-9aa6-88adb79c1959" width="400" height="250"/>
</p>

가장 아쉬웠던 부분은 저희 팀이 기획했던 아이디어가 이미 번개장터에서 그리고 다른 메이저 중고거래 플랫폼에서 기능으로 구현이 되어있다는 점이었습니다. 예를 들어, 유저가 상품 이미지를 올린 후 상품명을 입력할 시 자동으로 3가지의 카테고리가 나오는 기능이 구현되어 있었습니다 (번개장터 혹은 당근마켓 앱을 사용하신다면 더 자세히 알아볼 수 있습니다!). 해결이 필요한 문제를 풀기위해 해당 프로젝트를 기획했었기에 아쉬움이 남았습니다. 이 경험을 통해 사전조사를 철저히 하는 것의 중요성을 느낄 수 있었습니다. 

<p align="center">
<img src="https://github.com/owenchoi96/templates/assets/123911225/16b30f60-556f-4e74-9641-4fbc8f400573" width="400" height="250"/>
</p>

두번째는 프로젝트를 진행한 프로세스에 아쉬움이 있었습니다. 이 부분은 국비지원 교육이 끝나고 개인적으로 공부를 하면서 깨달았던 부분인데, 프로젝트를 진행할 때 분석의 순서를 따르지 않았던 것을 발견했습니다. 사실 단순히 딥러닝 모델을 공부하고 적용하는 것이 이번 프로젝트의 목적 및 목표였다면 아쉬움이 없었을 것 같습니다. 단순히 모델링의 성능에 대한 부분을 개선하지 못했다는 등의 아쉬움만 남았을 것 같습니다. 

하지만 얼마전 [빅데이터 시대, 성과를 이끌어내는 데이터 문해력](https://product.kyobobook.co.kr/detail/S000001019698)이라는 책을 읽고나서 프로젝트의 진행과정에 문제가 있음을 발견했습니다. 첫번째로 문제 및 목적을 정의한 후에 **현재 상태를 비교 및 분석**하는 과정이 생략되어 있다는 것을 발견했습니다. 이번 프로젝트를 진행하기 전에 당근마켓에서 상품 카테고리를 잘못선택하는 비율이 20%정도 된다는 뉴스기사를 읽은 적이 있었습니다. 그래서 이 문제를 해결하기 위해 이번 프로젝트를 진행했지만, 막상 **어떤 기준으로 얼마나 실제 상품이 잘못 분류되고 있는지에 대한 분석**은 진행을 하지 않았다는 것을 발견했습니다. 얼마나 잘못 분류되고 있는지를 알아야 저희가 모델링을 했을때 얼만큼 개선을 했다라는 before/after 과정을 나타낼 수 없었습니다. 때문에 저희가 분류 모델을 개발했다고 하더라도 이 모델을 왜 써야하는지, 과연 믿을 수 있는것인지에 대한 대답을 하기 어려울 수 있다는 것을 느꼈습니다. 

또한 **해당 프로젝트의 타겟 대상이 고객인지 아니면 상품 카테고리를 관리하는 현업가인지**에 대한 정의도 내리지 않았다는 것을 발견했습니다. 만약 제품을 사용하는 유저를 위한 기획이었다면, 상품명에 기반한 카테고리 3개를 추천하여 고객이 상품의 카테고리를 빠르고 쉽게 선택하는 것이 적절할 수 있다는 생각을 했습니다. 하지만 제품 분류의 정확도를 중요시여기는 관리자라면 최대한 한 개의 카테고리를 정확하게 예측하는 모델이 필요했을 수 있다는 것을 느꼈습니다. 이것은 단순히 저의 생각이고 테스트해볼 가설이지만, 이처럼 개발을 통한 기능 구현을 했을 때 **실제 사용자**가 누구인지에 대한 명확한 정의를 하지 않는다면 방향성이 완전히 달라질 수 있다는 것을 느꼈습니다. 

<p align="center">
<img src="https://github.com/owenchoi96/templates/assets/123911225/d2ad6dfb-cd22-46ec-a874-5755162e16ae" width="400" height="250"/>
</p>

또한 아쉬웠던 점은 왜 앙상블 모델을 최종 목표로 설정했을까였습니다. 기획 초기 분류 모델 관련 아티클들을 보며 네이버 쇼핑팀에서 이미지, 상품명, 상품정보 등등을 활용하여 카테고리에 따라 상품 분류를 효과적으로 한다는 것을 읽었습니다. 그래서 저희 팀은 중고 상품의 경우 네이버 쇼핑의 상품명보다 더 지저분(?) 할 확률이 높으니 앙상블 모델을 사용하여 분류 정확도를 높일 수 있을 것이라 예상했습니다. 하지만 텍스트 모델 만을 사용했을 때 성능이 충분히 좋다는 것을 발견할 수 있었고 앙상블 모델을 사용하더라도 성능이 크게 오르지 않는다는 것을 발견했습니다. 아마 그래서인지 번개장터도 이미지와 상품명을 넣었을 때 카테고리를 추천해주는 것이 아닌 상품명만 넣었을 때 카테고리를 추천해주는 기능을 만들지 않았을까라는 생각도 했습니다. 만약 해당 기획을 하나의 가설로 보고 다른 가설들 혹은 모델링들은 어떨까 예상하며 빠르게 다른 모델들도 실험하고 최선의 선택은 무엇이였을까 고민해지 못한 것이 아쉬웠습니다. 

하지만 프로젝트 전체 기간을 프로젝트에만 온전히 보낸 것이 아니라 딥러닝 언어 모델 및 파이토치 프레임워크 공부에도 시간을 많이 빼겼습니다. 또한 데이터를 수집하는 과정에서 크롤링에도 오랜 시간이 걸렸고, 특히 이미지 데이터를 크롤링하는 것은 굉장히 큰 수요였습니다. 그래서 이번 프로젝트를 경험삼아 많이 배웠으니 다음 프로젝트에서는 더 발전되고 더 정교한 분석을 하고 싶습니다. 

