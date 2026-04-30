from engram.agent.react import AgentResult, AgentStep


def test_agent_step_round_trip():
    step = AgentStep(
        hop=1,
        tool_name="search_facts",
        tool_input={"query": "x"},
        tool_result="result text",
        elapsed_s=0.5,
    )
    assert step.hop == 1
    assert step.tool_name == "search_facts"
    assert step.tool_input == {"query": "x"}


def test_agent_result_round_trip():
    res = AgentResult(answer="42", abstained=False, n_hops=2)
    assert res.answer == "42"
    assert res.n_hops == 2
    assert res.trace == []


def test_agent_step_defaults():
    s = AgentStep(hop=3)
    assert s.tool_name is None
    assert s.tool_input == {}
    assert s.text_emitted is None
