# 예제 코드

챕터 단위 디렉토리로 관리합니다. 각 디렉토리는 해당 챕터의 노트북과 실행 중 생성되는 파일을 담습니다.

```
examples/
  ch05-human-in-the-loop/
    hitl_example.ipynb
  ch06-middleware/        (예정)
```

- 디렉토리명: `chNN-슬러그` (책 CHAPTER 번호 기준, 영문 kebab-case)
- 의존성은 프로젝트 루트의 `pyproject.toml` / `uv.lock`에서 공통 관리 (챕터별로 나누지 않음)
- 실행: 프로젝트 루트에서 `uv run jupyter lab examples/chNN-슬러그/파일명.ipynb`
