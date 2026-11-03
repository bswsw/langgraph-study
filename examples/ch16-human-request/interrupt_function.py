"""16장 HumanRequest 패턴 — interrupt() 버전 (책 외 참고 구현)

book_interrupt_before.py 와 같은 그래프에서 아래 세 곳만 바꿨다.
  ① human_node : 노드 "안"에서 interrupt() 로 멈추고, 재개 값으로 ToolMessage 를 직접 만든다
  ② compile    : interrupt_before 제거
                 (남겨 두면 노드 "앞"에서 먼저 멈추는데, 그 시점엔 interrupt() 가 아직 안 불려서
                  __interrupt__ 에 request 가 없다. stream(None) 으로 재개하면 interrupt() 에서 또 멈춘다)
  ③ 실행 코드   : get_state / update_state / stream(None) 대신 Command(resume=...) 로 재개

흐름
  1) 사용자 질문 → chatbot 이 HumanRequest 호출 → ask_human=True        (책과 동일)
  2) select_next_node 가 "human" 반환 → human_node 실행 시작              (책과 동일)
  3) human_node 안의 interrupt() 에서 멈춤 → invoke 결과에 __interrupt__ 로 request 가 담겨 나옴
  4) Command(resume="답변") 으로 재개 → human_node 가 "처음부터 다시" 실행
     → 이번엔 interrupt() 가 "답변" 을 반환 → ToolMessage 생성, 플래그 해제 → chatbot 최종 답
"""

from typing import Annotated

from typing_extensions import TypedDict
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
from langchain_teddynote.tools.tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command


# ───────────────────────── 상태 ─────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_human: bool  # 사람에게 질문할지 여부


# ───────────────────────── 신호용 도구 ─────────────────────────
class HumanRequest(BaseModel):
    """Forward the conversation to an expert. Use when you can't assist directly or the
    user needs assistance that exceeds your authority.
    To use this function, pass the user's 'request' so that an expert can provide
    appropriate guidance.
    """

    request: str


tool = TavilySearch(max_results=3)
tools = [tool, HumanRequest]  # LLM 에게는 둘 다 보여 준다

llm = init_chat_model("claude-sonnet-4-5")
llm_with_tools = llm.bind_tools(tools)


# ───────────────────────── chatbot 노드 ─────────────────────────
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])

    ask_human = False
    if response.tool_calls and response.tool_calls[0]["name"] == HumanRequest.__name__:
        ask_human = True  # ① HumanRequest 호출 → 플래그 ON

    return {"messages": [response], "ask_human": ask_human}


# ───────────────────────── human 노드 ─────────────────────────
def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    # 1번째 실행: interrupt() 에서 멈춘다 (아래 return 까지 가지 않음).
    # 2번째 실행(재개): 노드가 처음부터 다시 돌고, interrupt() 가 Command(resume=...) 값을 반환한다.
    ai_message = state["messages"][-1]
    request = ai_message.tool_calls[0]["args"]["request"]
    answer = interrupt({"request": request})  # 멈춤 / 재개 시 전문가 답변이 들어옴
    return {
        "messages": [create_response(answer or "No response from human.", ai_message)],
        "ask_human": False,  # 플래그 해제
    }


# ───────────────────────── 라우팅 ─────────────────────────
def select_next_node(state: State):
    if state["ask_human"]:
        return "human"  # ② 플래그가 켜져 있으면 human 으로
    return tools_condition(state)  # "tools" 또는 END


# ───────────────────────── 그래프 ─────────────────────────
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))  # HumanRequest 는 넣지 않는다
graph_builder.add_node("human", human_node)

graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(
    checkpointer=MemorySaver(),
    # interrupt_before 없음 — 멈추는 건 human_node 안의 interrupt()
)


# ───────────────────────── 실행 ─────────────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    user_input = "이 AI 에이전트를 구축하기 위해 전문가의 도움이 필요합니다. 도움을 요청할 수 있나요?"

    # [1] 첫 실행 → chatbot 이 HumanRequest 호출 → human_node 안 interrupt() 에서 멈춤
    for event in graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()

    snapshot = graph.get_state(config)
    print("next:", snapshot.next)  # ('human',)
    print("interrupt:", snapshot.tasks[0].interrupts[0].value)  # {'request': '...'}  ← 전문가에게 보여 줄 내용

    # [2] 전문가 답변은 "텍스트만" 넘긴다 — ToolMessage 생성은 human_node 가 한다
    human_response = (
        "전문가들이 도와드리겠습니다! 에이전트 구축을 위해 LangGraph를 확인해 보시기를 적극 추천드립니다. "
        "단순한 자율 에이전트보다 훨씬 더 안정적이고 확장성이 뛰어납니다. "
        "https://wikidocs.net/233785에서 더 많은 정보를 확인할 수 있습니다."
    )

    # [3] 재개 → human_node 재실행(interrupt() 가 답변 반환) → chatbot 최종 답
    for event in graph.stream(Command(resume=human_response), config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()
