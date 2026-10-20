from onecli.agent import Agent
from onecli.providers.base import Turn, ToolCall
from onecli import context

class Provider:
    def __init__(self, turns):
        self.turns=iter(turns)
        self.calls=[]
    def chat(self,messages,tools=None,**kwargs):
        self.calls.append((list(messages), tools))
        return next(self.turns)

def call(i):
    return Turn(tool_calls=[ToolCall(id=str(i),name='ls',arguments={})])

def test_last_step_delivers_findings_without_increasing_budget(tmp_path):
    p=Provider([call(1),call(2),Turn(text='Found two issues; implementation remains unfinished.')])
    result=Agent(p,str(tmp_path),max_steps=3,autocompact=False).run('Review the project')
    assert result.stopped=='answered'
    assert result.tool_calls==2
    assert len(p.calls)==3
    assert p.calls[-1][1] == p.calls[0][1]  # stable cached tool prefix
    assert 'final response step' in p.calls[-1][0][-1]['content']
    assert 'unfinished' in result.text

def test_model_cannot_execute_tools_on_reserved_step(tmp_path):
    p=Provider([call(1),Turn(tool_calls=[ToolCall(id='bad',name='write',arguments={'path':'should-not-exist','content':'oops'})])])
    result=Agent(p,str(tmp_path),max_steps=2,autocompact=False).run('Review only')
    assert not (tmp_path/'should-not-exist').exists()
    assert result.stopped=='budget'
    assert result.tool_calls==2  # requested calls remain measurable, only one executed
    assert len(result.steps[-1].tool_results)==0
    assert not context.orphaned_tool_results(result.messages)

def test_compactor_sees_the_request_it_preserves():
    history=[{'role':'system','content':'system'}, {'role':'user','content':'Research improvements to one cli'}]
    for i in range(6):
        history += [{'role':'assistant','content':'', 'tool_calls':[{'id':str(i),'type':'function','function':{'name':'ls','arguments':'{}'}}]}, {'role':'tool','name':'ls','tool_call_id':str(i),'content':'fixture'}]
    seen=[]
    new, report=context.compact(history,lambda head: seen.extend(head) or 'Research in progress',keep_last=4)
    assert report.ok
    # Since verbatim request propagation the request is carried across VERBATIM rather than
    # summarized, so it is deliberately not input to the summarizer. Feeding
    # it in was the bug that commit names: the run spent its turns summarizing
    # the task instead of doing it. What must still hold is that the request
    # survives compaction exactly once, unparaphrased.
    assert history[1] not in seen
    assert new.count(history[1])==1
