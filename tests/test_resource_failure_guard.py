import json
from onecli.loopguard import LoopGuard

def diagnostic(model='qwen',kind='model_load_failed'):
 return 'exit 2\nLOOK_DIAGNOSTIC '+json.dumps({'model':model,'endpoint':'localhost:11434','kind':kind,'http_status':500})

def test_different_commands_same_model_failure():
 g=LoopGuard()
 assert not g.observe('bash',{'command':'look image.png'},diagnostic())
 notes=g.observe('bash',{'command':'look other.png'},diagnostic())
 assert any('same resource failure' in s for s in notes)

def test_different_models_are_not_conflated():
 g=LoopGuard()
 g.observe('bash',{'command':'one'},diagnostic())
 assert not g.observe('bash',{'command':'two'},diagnostic('other'))

def test_successful_printed_diagnostic_not_a_failure():
 g=LoopGuard()
 for c in ['one','two']:
  assert not g.observe('bash',{'command':c},diagnostic().replace('exit 2','exit 0'))


