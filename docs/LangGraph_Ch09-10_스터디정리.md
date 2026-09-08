# 테디노트의 LangGraph 에이전트 비법노트
## 이번 주 스터디 범위: CHAPTER 09 ~ CHAPTER 10 요약

---

## 📘 CHAPTER 09. LangGraph 메모리 추가하기

### 01. MemorySaver 체크포인터
- 체크포인터(Checkpointer)는 그래프의 각 단계에서 상태를 저장하여, 이후 동일한 대화를 이어서 진행할 수 있게 하는 컴포넌트다.
- `MemorySaver`는 **인메모리** 체크포인터로, 개발·테스트 환경에 적합하다.
- 프로덕션 환경에서는 서버 재시작 시에도 상태가 유지되어야 하므로 `PostgresSaver`, `SqliteSaver` 같은 영구 저장소 기반 체크포인터를 사용해야 한다.

```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
```

### 02. 메모리가 있는 챗봇 구축하기
- 상태(State)의 `messages` 필드에 `add_messages` 리듀서를 적용하면 새 메시지가 기존 리스트를 덮어쓰지 않고 자동으로 누적된다.
- `graph_builder.compile(checkpointer=memory)`로 체크포인터와 함께 컴파일한다.

### 03. 멀티턴 대화 테스트
- `RunnableConfig`의 `configurable={"thread_id": ...}`로 대화 세션을 구분한다.
- **같은 thread_id** → 이전 대화 기억 (예: 이름을 알려주면 이후 질문에도 기억)
- **다른 thread_id** → 별도 세션으로 취급, 이전 대화 기억 못함 (카카오톡 채팅방처럼 독립적)
- 상태 조회: `graph.get_state(config)` → `values`(메시지), `config`(thread_id/checkpoint_id), `next`(다음 실행 노드)
- 이력 조회: `graph.get_state_history(config)`로 시간 역순 체크포인트 확인, 롤백/디버깅에 활용 가능

### 04. 도구와 메모리 결합하기
- TavilySearch 같은 검색 도구를 사용하는 에이전트에도 체크포인터를 적용하면, 도구 호출 결과까지 포함한 전체 대화 기록이 저장된다.
- 이전에 검색한 정보를 바탕으로 후속 질문("방금 내용을 요약해줘")에 답변 가능.

### 05~08. 단기 메모리 관리 전략 (컨텍스트 윈도우 & 비용 문제 해결)

| 전략 | 방법 | 특징 |
|---|---|---|
| **트리밍** (`trim_messages`) | 오래된 메시지를 잘라 최근 메시지만 LLM에 전달 | 상태(전체 기록)는 그대로 보존, LLM 호출 비용만 관리 |
| **삭제** (`RemoveMessage`) | 특정 메시지를 상태에서 영구 제거 | 저장 데이터량 감소, 민감정보 삭제에 유용 |
| **요약** (Summarization) | 오래된 메시지를 LLM으로 요약 → 시스템 메시지로 대체 후 원본 삭제 | 정보 손실 최소화하며 토큰 절약 |

- `trim_messages` 옵션: `strategy`("last"=최근 유지 / "first"=초반 유지), `max_tokens`, `token_counter`, `start_on`, `include_system`
  - ⚠️ `token_counter`에는 **LLM 모델 객체**를 전달해야 함. `len` 함수를 쓰면 메시지 개수를 토큰 수로 잘못 계산해 트리밍이 오작동한다.
- `RemoveMessage(id=...)`는 `update_state()`와 함께 사용해 특정 메시지를 선택 삭제. 그래프 노드로 만들면 매 실행마다 자동으로 오래된 메시지 정리 가능 (예: 최근 3개만 유지).
- 요약 전략 흐름: ① 오래된 메시지를 LLM으로 2~3문장 요약 → ② `SystemMessage`로 상태에 추가 → ③ 원본 메시지는 `RemoveMessage`로 삭제. 메시지가 일정 개수(예: 6개) 초과 시 자동 요약되도록 조건부 노드 구성.

### 09. 프로덕션 환경에서 영구 저장소 사용하기
- `MemorySaver`는 서버 재시작 시 데이터가 사라지므로, 프로덕션에서는 아래 체크포인터 사용:
  - **PostgresSaver**: ACID 트랜잭션 지원, 엔터프라이즈 환경 표준
  - **SqliteSaver**: 가벼운 로컬 저장소
  - **RedisSaver**: 초고속 인메모리 DB, 캐싱에 최적화
- `with PostgresSaver.from_conn_string(DB_URI) as checkpointer:` 컨텍스트 매니저로 연결 관리, 최초 `checkpointer.setup()`으로 테이블 생성.
- 새 연결(새 프로세스)에서도 같은 thread_id로 조회하면 이전 대화가 그대로 유지됨 — 이것이 영구 저장소를 써야 하는 핵심 이유.

---

## 📗 CHAPTER 10. LangGraph MCP 실습

> MCP(Model Context Protocol)는 애플리케이션이 언어 모델에 도구와 컨텍스트를 제공하는 방법을 표준화한 오픈 프로토콜. USB가 다양한 주변기기를 하나의 규격으로 연결하듯, MCP는 다양한 서비스·도구를 일관된 방식으로 LLM에 연결한다.

### 01~03. 패키지 설치 & 서버/클라이언트 설정
- `langchain-mcp-adapters`: LangChain 에이전트가 MCP 서버 도구를 쓸 수 있게 해주는 어댑터 라이브러리
- `MultiServerMCPClient`: 여러 MCP 서버를 동시에 관리·연결, 도구들을 하나의 목록으로 통합
- 전송 방식 2가지:
  - **stdio**: 클라이언트가 서버를 서브프로세스로 직접 실행, 로컬 개발에 편리 (서버를 따로 띄울 필요 없음)
  - **streamable_http**: 서버가 독립 프로세스로 실행 중이어야 함, 원격/프로덕션 환경에 적합
- 실습 서버 3종: `mcp_server_local.py`(날씨, stdio), `mcp_server_remote.py`(시간, HTTP), `mcp_server_rag.py`(PDF 검색 RAG, stdio)
- `MCP Inspector`: 브라우저에서 서버 도구 목록 확인·직접 호출해볼 수 있는 디버깅 도구

### 04. 에이전트와 MCP 통합하기
- `create_agent()` (LangChain 1.0+ 권장, 기존 `langgraph.prebuilt.create_react_agent` 대체)로 MCP 도구 기반 에이전트 생성
- 날씨(stdio) + 시간(HTTP) 서버를 동시에 연결해 하나의 에이전트가 여러 서버 도구 사용 가능
- 같은 thread_id로 연속 질문 시 대화 컨텍스트 유지되어 자연스럽게 응답
- **RAG MCP 서버** 예제: `retrieve` 도구로 PDF 문서에서 정보 검색 → "삼성 가우스", "구글의 Anthropic 투자액" 등 질의응답

### 05. ToolNode와 MCP 통합하기
- `create_agent` 대신 `StateGraph` + `ToolNode` + `tools_condition`으로 그래프의 각 노드를 직접 정의 → 더 세밀한 제어 가능
- 장점: ① 세밀한 제어(노드별 커스텀 로직/오류처리) ② 유연한 워크플로(조건부 분기) ③ 확장성(MCP 도구 + 일반 LangChain 도구 혼합 사용)
- 예: MCP 시간 조회 도구 + Tavily 뉴스 검색 도구를 조합해 "시간 조회 후 해당 날짜 뉴스 검색" 같은 복합 작업을 자동 수행

### 06. 외부 MCP 서버에서 서드파티 도구 사용하기
- **Context7 MCP 서버**: 최신 프로그래밍 언어/프레임워크 공식 문서를 실시간 검색 (`resolve-library-id`, `query-docs` 도구)
- LLM 학습 데이터 컷오프 이후 변경된 API 패턴도 최신 문서 기반으로 정확히 참조 가능
- `npx`로 직접 실행, stdio 전송 방식 사용. Smithery 같은 MCP 서버 레지스트리에서 다양한 서드파티 서버 검색 가능
- 실습 예: Context7로 최신 LangGraph 문서에서 ReAct Agent 패턴 검색 → 그 정보 바탕으로 Tavily 검색 기능 포함된 ReAct Agent 코드를 자동 생성하는 복합 작업

---

### 💡 두 챕터를 관통하는 핵심 흐름
CH09(메모리 관리) → CH10(외부 도구 연결)으로 이어지며, **"대화 상태를 어떻게 유지·관리할 것인가"**에서 **"에이전트가 외부 세계와 어떻게 상호작용할 것인가"**로 주제가 확장된다. 다음 PART 04(멀티 에이전트)에서는 이 둘을 조합해 여러 에이전트가 역할을 나눠 협업하는 구조를 다룬다.




&nbsp;