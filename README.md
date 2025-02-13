# kmleMaster

## 0. 데이터 수집
- **소스:** 의사 국가고시, 임상종합의학평가 문제 (PDF 형식)
- **처리 방식:** PDF 전처리 → OCR 작업 수행

---

## 1. [유사 문제 검색](https://github.com/1000century/kmleMaster/blob/main/model)
### 목표
- 토큰 기반 **유사 문제 검색 알고리즘 개발**
- **유사도 검색 알고리즘:** `rank_bm25`
- **토크나이저:** `bert-base-multilingual-cased`
- **검색 범위:** 각 문제의 `text` 부분으로 한정
- **출력 결과 설정:** `Top 2`

### 1-1. 토크나이저 분석 결과
#### 특정 단어 포함 여부 확인
- `남아` → `남` + `아` 형태보다는 `남아` 그대로 토큰화되는 것이 이상적
- 너무 크거나 작지 않은 단위로 토크나이징 되는 **적절한 토크나이저 선택**이 중요
- 자주 등장하는 주요 단어들 중심으로 분석

![토크나이저 분석](https://github.com/user-attachments/assets/7f688e0f-8d3f-4e0c-80bd-e51e0407796d)

#### 토크나이저로 분할된 토큰 개수 분석
![토큰 개수 분석](https://github.com/user-attachments/assets/f21942c6-530d-4aac-85a8-9b9c061509f1)

---
