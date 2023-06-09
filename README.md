# level2_cv_datacentric-cv-05
level2_cv_datacentric-cv-05 created by GitHub Classroom

## pre-commit
pip install pre-commit<br>
git clone 진행한 directory로 이동<br>
pre-commit install<br>
코드 정상 실행된 경우 git add, commit, push 진행

--------------

## Commit Type
- feat : 새로운 기능 추가, 기존의 기능을 요구 사항에 맞추어 수정
- fix : 기능에 대한 버그 수정
- build : 빌드 관련 수정
- chore : 패키지 매니저 수정, 그 외 기타 수정 ex) .gitignore
- ci : CI 관련 설정 수정
- docs : 문서(주석) 수정
- style : 코드 스타일, 포맷팅에 대한 수정
- refactor : 기능의 변화가 아닌 코드 리팩터링 ex) 변수 이름 변경
- test : 테스트 코드 추가/수정
- release : 버전 릴리즈

--------------

## train.py argparse 추가 내역  
- --wandb_project : wandb project 이름
- --wandb_name : wandb project
- --my_opt : optimizer 선택 <Adam(default), SGD, RMSprop, Adagrad, AdamW>
- --my_sched : scheduler 선택 <MultiStepLR(default), CosineAnnealingLR, ReduceLROnPlateau>
- --factor : 학습률 조절에 사용, new lr = lr * factor로 구한다
- --patience : 학습률 조절 시점 설정에 사용, ReduceLROnPlateau scheduler사용 중 loss값의 update가 patience만큼 없으면 lr 개선
- --milestones : MultiStepLR scheduler를 사용할 때 학습률 변경 지점을 list 형식으로 대입
