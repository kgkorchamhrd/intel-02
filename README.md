# 상공회의소 경기인력개발원 인텔교육 2기

## Clone code 

```shell
git clone --recurse-submodules https://github.com/kgkorchamhrd/intel-02.git
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

## Preparation

### Git LFS(Large File System)

* 크기가 큰 바이너리 파일들은 LFS로 관리됩니다.

* git-lfs 설치 전

```shell
# Note bin size is 132 bytes before LFS pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

* git-lfs 설치 후, 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
$ sudo apt install git-lfs

$ git lfs pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)

## Team projects

### 제출방법

1. 팀구성 및 프로젝트 세부 논의 후, 각 팀은 프로젝트 진행을 위한 Github repository 생성

2. [doc/project/README.md](./doc/project/README.md)을 각 팀이 생성한 repository의 main README.md로 복사 후 팀 프로젝트에 맞게 수정 활용

3. 과제 제출시 `인텔교육 2기 Github repository`에 `New Issue` 생성. 생성된 Issue에 하기 내용 포함되어야 함.

    * Team name : Project Name
    * Project 소개
    * 팀원 및 팀원 역활
    * Project Github repository
    * Project 발표자료 업로드

4. 강사가 생성한 `Milestone`에 생성된 Issue에 추가 

### 평가방법

* [assessment-criteria.pdf](./doc/project/assessment-criteria.pdf) 참고

### 제출현황

### Team: 뭔가 센스있는 팀명
<프로젝트 요약>
* Members
  | Name | Role |
  |----|----|
  | 채치수 | Project lead, 프로젝트를 총괄하고 망하면 책임진다. |
  | 송태섭 | Project manager, 마일스톤을 생성하고 프로젝트 이슈 진행상황을 관리한다. |
  | 정대만 | UI design, 사용자 인터페이스를 정의하고 구현한다. |
  | 채소연 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 권준호 | Architect, 프로젝트의 component를 구성하고 상위 디자인을 책임진다. |
* Project Github : https://github.com/goodsense/project_awesome.git
* 발표자료 : https://github.com/goodsense/project_aewsome/doc/slide.ppt


### ~~Team: 100만 버튜버~~  
~~<프로젝트 요약>~~
~~Pose Estimate, Face Tracking, Voice Changer를 이용하여 Virtual Character를 만드는 프로그램~~  
~~* Members~~  
  | Name | Role |  
  |----|----|  
  | ~~김정문~~ | ~~Project leader, 프로젝트를 총괄하고 망하면 책임진다.~~ |  
  | ~~신현택~~ | ~~Projecy Member, 리더와 매니저의 스케쥴 관리를 잘 따라 프로젝트를 진행한다.~~ |  
  | ~~임민우~~ | ~~Project Manager, 프로젝트의 진행 상황을 관리한다.~~ |  

~~* Project Github : https://github.com/JEONGMOONKIM/VR_C~~  
~~* 발표자료 : https://github.com/kgkorchamhrd/intel-02/blob/main/doc/project/team1/presentation.ppt~~  



### Team name : cp&mv
<프로젝트 요약>  
  OpenPose를 이용하여 원하는 동작을 따라하는 프로그램
* Members
  | Name | Role |
  |----|----|
  | 김우석 | Project member, 프로젝트 진행에 도움이 된다. |
  | 신현택 | Project Leader, 프로젝트를 총괄한다. |
  | 엄재식 | Project manager, 프로젝트 이슈 진행상황을 관리한다. |
  | 허환욱 | Project member, 리더와 매니저의 말을 잘 따른다. |
  
* Project Github : https://github.com/BrotherHwan/Final_project.git
* 발표자료 : https://github.com/BrotherHwan/Final_project/blob/main/cp%26mv(final_project).pptx


### Team name : Coco_Friend
<프로젝트 요약>  
  얼굴인식 : OpenCV(모델 빌드 및 학습), 대화 : LLM(챗봇,STT,TTS) 를 활용하여 맞벌이 부부 자녀 케어 AI
* Members
  | Name | Role |
  |----|----|
  | 허진호 | Project Lead, 리더와 매니저의 말을 잘 따른다. |
  | 이가원 | Project manager, 리더와 매니저의 말을 잘 따른다. |
  | 박도월 | Project member, 프로젝트 이슈 진행상황을 관리한다. |
  | ~~이동제~~ | ~~Project lead, 프로젝트를 총괄한다.~~ |
  | ~~손영빈~~ | ~~Project member, 리더와 매니저의 말을 잘 따른다.~~ |
  
  
* Project Github : https://github.com/jinhoheoo/coco-friend-project
* 발표자료 : https://github.com/jinhoheoo/coco-friend-project/blob/main/03_coco_friend.pptx


### Team name : Vision Amateur
<프로젝트 요약>
Gesture Recognition, Image Classification & Localization, Speech 2 Text, Translation 및 정보검색을 통해 VR UX 기술 구현
* Members
  | Name | Role |
  |----|----|
  | 김마로 | Project Leader, 프로젝트를 총괄 및 검수. |
  | 이원희 | Project Manager, 진행상황 점검 및 UX Design. |
  | 임우섭 | AI modeling, Vision 및 Gesture 인식 관련 AI Model 구현. |
  | 오흥천 | Text translation, 자연어 처리 및 정보 검색 Algorithm Modeling. |
  
* Project Github : https://github.com/ghkfkd1/04_Travel_Helper.git
* 발표자료 : https://github.com/ghkfkd1/04_Travel_Helper/blob/main/miniproject.odp



### Team: Xtreme World
<프로젝트 요약> 양파전 프로스포츠에서 적절한 양의 원정석을 지정
* Members
  | Name | Role |
  |----|----|
  |<span style="color:#808080"> 송기찬 </span>|<span style="color:#808080"> Head, 기본 뼈대 및 코드 제작 </span>|
  | 안효재 | Project Member, 트레이닝 시각 자료 수집 |
  |<span style="color:#808080"> 김정진 </span>|<span style="color:#808080"> Project Member </span>|
* Project Github : https://github.com/MobilSuper/team5proj.git
* 발표자료 : https://github.com/MobilSuper/team5proj/doc/team5slide.ppt

