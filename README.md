# kmleMaster

## 0. 데이터 수집
- 의사 국가고시, 임상종합의학평가 문제의 PDF파일
- 방식: PDF 전처리 후 OCR 작업 수행

## 1. 유사 문제 검색 [https://github.com/1000century/kmleMaster/model]
- 유사도 검색 알고리즘: rank_bm25
- 토크나이저: bert-base-multilingual-cased
- 사용 범위: 각 문제의 'text' 부분으로 한정
- 출력 결과 설정: Top2

### 1-1. 토크나이저 분석결과
- 특정 단어 포함 여부 확인
  `남아` -> `남` + `아` 와 같은 방식보다는 `남아`가 바로 나오는 방식이 좋다.
   <br>따라서 너무 크지도 않고 작지도 않은 단위로 토크나이징 되는 토크나이저를 찾는 것이 목표이다.
  <br>자주 등장하는 주요한 단어들로 선정하였다.
  ![image](https://github.com/user-attachments/assets/7f688e0f-8d3f-4e0c-80bd-e51e0407796d)
- 토크나이저로 쪼개진 토큰 개수
  ![image](https://github.com/user-attachments/assets/f21942c6-530d-4aac-85a8-9b9c061509f1)


