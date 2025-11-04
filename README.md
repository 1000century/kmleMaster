# kmleMaster

## 0. 데이터 수집
- **소스:** 의사 국가고시, 임상종합의학평가 문제 (PDF 형식)
- **처리 방식:** PDF 전처리 → OCR 작업 수행
- **사용 목적:** 본 데이터는 오직 **개인 학습용**으로만 사용하기 위해 수집됨

---

## 1. [유사 문제 검색 알고리즘](https://github.com/1000century/kmleMaster/blob/main/model)

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/4e3c369b-6f1b-459b-8f69-2130583f752e"></td>
    <td><img src="https://github.com/user-attachments/assets/95ff0d00-44f3-4df3-aa45-5ee7af82917d"></td>
  </tr>
</table>

### 목표
- 토큰 기반 **유사 문제 검색 알고리즘 개발**
- **유사도 검색 알고리즘:** `rank_bm25`
- **토크나이저:** `LeverageX/finbert-wechsel-korean`
- **검색 범위:** 각 문제의 `text` 부분으로 한정
- **출력 결과 설정:** `Top 2`

### 1-1. 토크나이저 분석 결과
#### 특정 단어 포함 여부 확인
- `남아` → `남` + `아` 형태보다는 `남아` 그대로 토큰화되는 것이 이상적
- 너무 크거나 작지 않은 단위로 토크나이징 되는 **적절한 토크나이저 선택**이 중요
- 자주 등장하는 주요 단어들 중심으로 분석
#### 토크나이저로 분할된 토큰 개수 분석
<table>
  <td><img src="https://github.com/user-attachments/assets/7f688e0f-8d3f-4e0c-80bd-e51e0407796d"</td>
  <td><img src="https://github.com/user-attachments/assets/f21942c6-530d-4aac-85a8-9b9c061509f1"</td>
</table>



### 기타
- 비커 css
```css
/* ============================================
   비커 시각화 핵심 CSS 컴포넌트
   용도: 농도/수치 비교를 시각적으로 표현
   ============================================ */

/* 1. 비커 컨테이너 레이아웃 */
.water-visual {
    display: flex;
    gap: 25px;                      /* 비커 간 간격 */
    justify-content: center;
    margin: 25px 0;
    flex-wrap: wrap;                /* 반응형 래핑 */
}

/* 2. 개별 비커 래퍼 */
.beaker {
    text-align: center;
    flex: 1;
    min-width: 180px;               /* 최소 너비 */
    max-width: 240px;               /* 최대 너비 */
}

/* 3. 비커 제목 (정상/비정상 구분) */
.beaker-title {
    font-size: 15px;
    font-weight: bold;
    margin-bottom: 12px;
    color: #2c3e50;
    min-height: 38px;               /* 높이 고정으로 정렬 */
    display: flex;
    align-items: center;
    justify-content: center;
}

.beaker.abnormal .beaker-title {
    color: #e74c3c;                 /* 비정상은 빨간색 */
}

/* 4. 비커 본체 (실린더 모양) */
.beaker-container {
    width: 180px;                   /* 비커 너비 */
    height: 240px;                  /* 비커 높이 */
    margin: 0 auto;
    border: 4px solid #34495e;      /* 기본 테두리 */
    border-radius: 0 0 18px 18px;  /* 하단만 둥글게 */
    position: relative;             /* 내부 요소 절대 위치 기준 */
    background: linear-gradient(to bottom, #ecf0f1 0%, #bdc3c7 100%);
    overflow: hidden;               /* 넘치는 내용물 숨김 */
}

/* 5. 정상/비정상 비커 스타일 */
.beaker.normal .beaker-container {
    border-color: #27ae60;          /* 초록 테두리 */
    box-shadow: 0 0 0 3px rgba(39, 174, 96, 0.2);  /* 초록 그림자 */
}

.beaker.abnormal .beaker-container {
    border-color: #e74c3c;          /* 빨강 테두리 */
    box-shadow: 0 0 0 3px rgba(231, 76, 60, 0.2);  /* 빨강 그림자 */
}

/* 6. 특정 비커 배경색 (선택사항) */
.beaker.blood-normal .beaker-container {
    background: linear-gradient(to bottom, #fff 0%, #e8f4f8 100%);
}

.beaker.blood-diluted .beaker-container {
    background: linear-gradient(to bottom, #fff 0%, #d5ebf5 100%);
}

/* 7. 액체 수위 (핵심!) */
.water-level {
    position: absolute;
    bottom: 0;                      /* 하단부터 시작 */
    width: 100%;
    background: linear-gradient(to top, #3498db, #5dade2);  /* 기본 물색 */
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 13px;
    transition: height 0.5s;        /* 높이 변화 애니메이션 */
}

/* 8. 액체 종류별 색상 & 높이 */
.beaker.blood-normal .water-level {
    height: 50%;                    /* 높이 = 농도 */
    background: linear-gradient(to top, #e74c3c, #ec7063);  /* 빨간색 */
}

.beaker.blood-diluted .water-level {
    height: 75%;                    /* 더 높음 = 희석됨 */
    background: linear-gradient(to top, #ffcccb, #ff9999);  /* 연한 빨강 */
}

.beaker.urine-normal .water-level {
    height: 40%;
    background: linear-gradient(to top, #f1c40f, #f8e08e);  /* 노란색 */
}

.beaker.urine-concentrated .water-level {
    height: 30%;                    /* 더 낮음 = 농축됨 */
    background: linear-gradient(to top, #d4a017, #f4a460);  /* 진한 노랑 */
}

/* 9. 입자 컨테이너 (농도 표현) */
.particles {
    position: absolute;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
}

/* 10. 개별 입자 */
.particle {
    position: absolute;
    width: 8px;
    height: 8px;
    background: #2c3e50;            /* 입자 색상 */
    border-radius: 50%;             /* 원형 */
    animation: float 3s infinite;   /* 떠다니는 효과 */
}

/* 입자 위치는 인라인으로: style="left: 20%; top: 30%;" */

/* 11. 입자 떠다니는 애니메이션 */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

/* 12. 비커 하단 수치 레이블 */
.beaker-label {
    margin-top: 12px;
    padding: 10px;
    background: white;
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.7;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* 13. 정상/비정상 레이블 스타일 */
.beaker.normal .beaker-label {
    border: 2px solid #27ae60;      /* 초록 테두리 */
    background: #e8f8f5;            /* 연한 초록 배경 */
}

.beaker.abnormal .beaker-label {
    border: 2px solid #e74c3c;      /* 빨강 테두리 */
    background: #ffe6e6;            /* 연한 빨강 배경 */
}

/* 14. 레이블 내 수치 강조 */
.beaker-label-value {
    font-size: 20px;
    font-weight: bold;
    margin: 4px 0;
}

/* 15. 수치 색상 (높음/낮음/정상) */
.concentration-high {
    color: #e74c3c;                 /* 빨강 = 높음 */
    font-weight: bold;
}

.concentration-low {
    color: #3498db;                 /* 파랑 = 낮음 */
    font-weight: bold;
}

.concentration-normal {
    color: #27ae60;                 /* 초록 = 정상 */
    font-weight: bold;
}

/* 16. 상태 배지 (선택사항) */
.status-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: bold;
    margin-top: 6px;
}

.status-normal {
    background: #27ae60;
    color: white;
}

.status-abnormal {
    background: #e74c3c;
    color: white;
}
```
