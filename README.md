# 건설 현장 안전고리 체결 판별 시스템

건설 현장 영상에서 작업자 주변의 안전고리를 찾고, 구조물 체결 여부를 분류하는 딥러닝 프로젝트의 실험 및 통합 추론 코드입니다. 객체 탐지·인스턴스 분할 모델로 안전고리 후보 영역을 검출하고, EfficientNet 또는 ViT 분류기로 `connected`와 `unconnected` 상태를 판별합니다.

문서, 회의록, 보고서와 발표 자료는 [docs 저장소](https://github.com/Tacademy-DL-team1/docs)에서 별도로 관리합니다.

## 코드 흐름

1. 이미지에서 안전고리와 주변 맥락을 crop합니다.
2. YOLO 또는 RF-DETR로 안전고리 후보 영역을 검출·분할합니다.
3. EfficientNet 또는 ViT로 체결 여부를 분류합니다.
4. ByteTrack과 상태 판정 로직을 결합해 영상 단위 결과를 생성합니다.

## 저장소 구조

```text
.
├── notebooks/
│   ├── preprocessing/   # crop 전처리 실험
│   ├── detection/       # YOLO·RF-DETR 탐지/분할 실험
│   ├── classification/  # EfficientNet·ViT 분류 실험
│   └── pipelines/       # 최종 통합 파이프라인 노트북
├── src/
│   ├── preprocessing/   # 재사용 가능한 crop 스크립트
│   └── pipelines/       # 영상 추론 및 최종 통합 스크립트
├── data/                # 로컬 데이터(업로드 금지, README만 추적)
├── models/              # 모델 가중치(업로드 금지, README만 추적)
├── outputs/             # 생성 결과(업로드 금지, README만 추적)
├── samples/             # 이미지·영상 샘플(업로드 금지, README만 추적)
├── FILE_GUIDE.txt       # 파일별 내용, 작성자, 노션 순서
└── requirements.txt
```

## 실행 환경

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Roboflow를 사용하는 노트북은 API 키를 코드에 적지 않고 환경변수로 설정합니다.

```bash
# PowerShell
$env:ROBOFLOW_API_KEY="your-key"
```

데이터·가중치 경로는 각 코드의 설정 셀 또는 `CONSTRUCTION_SAFETY_PROJECT_ROOT` 환경변수로 지정합니다. 노트북은 결과 출력과 실행 번호를 제거한 상태로 보관하므로 위에서 아래로 다시 실행해야 합니다.

## 저장소 정책

- 원본/가공 데이터, 이미지, 영상, 모델 가중치, 학습·추론 결과는 GitHub에 올리지 않습니다.
- API 키, 토큰, 계정 정보와 개인 경로를 커밋하지 않습니다.
- 공개가 필요한 대용량 파일은 저장소에 바로 추가하지 않고 팀 합의 후 외부 스토리지 또는 Git LFS 정책을 정합니다.
- 원작성자 정보는 `FILE_GUIDE.txt`, 코드 머리말, 이슈 및 PR 본문에 남깁니다. Git 커밋 작성자는 실제 커밋을 수행한 계정을 사용합니다.

## 유의 사항

이 코드는 연구·프로토타입 목적입니다. 실제 현장 안전 판단을 대체하지 않으며, 배포 전 별도의 현장 검증과 개인정보·영상정보 처리 검토가 필요합니다.

