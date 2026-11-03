"""16장 HumanRequest 패턴 — 책 예제 (정적 인터럽트: interrupt_before)

08-LangGraph-State-Customization.ipynb 의 코드를 순서대로 한 파일로 모았다.
interrupt_function.py 와 나란히 diff 해서 보면 바뀌는 곳만 드러나도록 구조를 맞췄다.

흐름
  1) 사용자 질문 → chatbot 이 HumanRequest 호출 → ask_human=True
  2) select_next_node 가 "human" 반환 → interrupt_before=["human"] 이라 human 노드 실행 "전"에 멈춤
  3) 바깥 코드가 전문가 답변을 ToolMessage 로 만들어 update_state 로 상태에 주입
  4) stream(None) 으로 재개 → human_node 는 이미 ToolMessage 가 있으니 플래그만 끔 → chatbot 최종 답
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
    # 이 노드에 들어오기 "전"에 이미 멈췄다가 재개된 상태.
    # 바깥에서 update_state 로 ToolMessage 를 넣었다면 그대로 두고,
    # 안 넣었다면 기본 응답을 채운다 (방어 로직).
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        "messages": new_messages,
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
    interrupt_before=["human"],  # human 노드 실행 "전"에 멈춤
)


# ───────────────────────── 실행 ─────────────────────────
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    user_input = "이 AI 에이전트를 구축하기 위해 전문가의 도움이 필요합니다. 도움을 요청할 수 있나요?"

    # [1] 첫 실행 → chatbot 이 HumanRequest 호출 → human 직전에서 멈춤
    for event in graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()

    snapshot = graph.get_state(config)
    print("next:", snapshot.next)  # ('human',)

    # [2] 전문가 답변을 "바깥에서" ToolMessage 로 만들어 상태에 주입
    ai_message = snapshot.values["messages"][-1]
    human_response = (
        "전문가들이 도와드리겠습니다! 에이전트 구축을 위해 LangGraph를 확인해 보시기를 적극 추천드립니다. "
        "단순한 자율 에이전트보다 훨씬 더 안정적이고 확장성이 뛰어납니다. "
        "https://wikidocs.net/233785에서 더 많은 정보를 확인할 수 있습니다."
    )
    tool_message = create_response(human_response, ai_message)
    graph.update_state(config, {"messages": [tool_message]})

    # [3] 재개 (입력 None) → human_node(플래그만 해제) → chatbot 최종 답
    for event in graph.stream(None, config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()
