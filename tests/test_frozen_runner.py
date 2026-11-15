import json
import pytest
from langchain_core.messages import AIMessage
from scripts.run_frozen_benchmark import Recorder, normalize_grade, judge_prompt
from scripts.verify_mapevolve_freeze import load_frozen


def test_recorder_preserves_reply_and_redacts_credentials(tmp_path, monkeypatch):
    from scripts import run_frozen_benchmark as runner
    monkeypatch.setattr(runner,'SECRETS',['secret-test-key'])
    class Client:
        def invoke(self, request):
            return AIMessage(content='answer',response_metadata={'provider':'fake'},usage_metadata={'input_tokens':1,'output_tokens':2,'total_tokens':3})
    Recorder(Client(),tmp_path).invoke({'question':'secret-test-key'})
    assert 'secret-test-key' not in (tmp_path/'request_1.json').read_text()
    reply=json.loads((tmp_path/'response_1.json').read_text())['response']
    assert reply['usage_metadata']['total_tokens']==3
    assert reply['response_metadata']['provider']=='fake'


def test_recorder_checkpoints_error(tmp_path):
    class Client:
        def invoke(self, request): raise TimeoutError('timeout')
    with pytest.raises(TimeoutError): Recorder(Client(),tmp_path).invoke({'question':'q'})
    assert (tmp_path/'request_1.json').exists()
    assert json.loads((tmp_path/'error_1.json').read_text())['exception_type']=='TimeoutError'


def test_judge_has_reference_but_no_model_identity():
    _,_,cases,_=load_frozen()
    prompt=judge_prompt(cases[0],'An answer')
    assert cases[0].reference['answer'] in prompt
    assert '<<<' not in prompt
    assert 'GEMMA3_4B' not in prompt


def test_unsupported_claim_cannot_receive_full_credit():
    assert normalize_grade({'score':4,'unsupported_claim':True})['score']==3
    with pytest.raises(ValueError): normalize_grade({'score':5})


def test_gpt_requests_are_paced(tmp_path, monkeypatch):
    from scripts import run_frozen_benchmark as runner
    delays=[]
    monkeypatch.setattr(runner,'GPT_NEXT_CALL',0.0)
    monkeypatch.setattr(runner.time,'monotonic',lambda:100.0)
    monkeypatch.setattr(runner.time,'sleep',delays.append)
    class Client:
        model_name='openai/gpt-5.1'
        def invoke(self, request): return {'answer':'ok'}
    recorder=Recorder(Client(),tmp_path)
    recorder.invoke({});recorder.invoke({})
    assert delays==[4.0]
