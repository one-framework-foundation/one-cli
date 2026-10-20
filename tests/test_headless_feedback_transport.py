import json
import socket
import pytest
from onecli.turn_feedback import TurnFeedback


def pair():
    parent, child = socket.socketpair()
    parent.settimeout(.2)
    return parent, TurnFeedback(child.detach())


def frame(ident, text):
    return json.dumps({'id':ident,'text':text}).encode()+b'\n'


def test_fragment_waits_until_complete_and_preserves_fifo():
    parent, child = pair()
    history = []
    try:
        payload=frame('a','purple')
        parent.sendall(payload[:7])
        assert not child.consume(history)
        assert history == []
        parent.sendall(payload[7:]+frame('b','also add a title'))
        assert child.consume(history)
        assert [m['content'] for m in history]==['purple','also add a title']
        receipts=parent.makefile('rb')
        assert json.loads(receipts.readline())=={'id':'a','accepted':True}
        assert json.loads(receipts.readline())=={'id':'b','accepted':True}
    finally:
        child.close()
        parent.close()


def test_eof_consumes_complete_frames_before_closing():
    parent, child = pair()
    history = []
    try:
        parent.sendall(frame("a", "purple"))
        parent.shutdown(socket.SHUT_WR)
        assert child.consume(history)
        assert history==[{'role':'user','content':'purple'}]
        assert json.loads(parent.recv(1024))['accepted'] is True
        assert child.closed
    finally:
        child.close()
        parent.close()


def test_eof_discards_incomplete_frame_and_closes():
    parent, child = pair()
    history = []
    try:
        parent.sendall(b'{"id":"a","text":"unfinished')
        parent.shutdown(socket.SHUT_WR)
        assert not child.consume(history)
        assert history==[]
        assert child.closed
    finally:
        child.close()
        parent.close()


@pytest.mark.parametrize('bad',[b'not json\n',b'[]\n',b'{"id":1,"text":"x"}\n',b'\xff\n'])
def test_invalid_frame_does_not_drop_following_valid_input(bad):
    parent, child = pair()
    history = []
    try:
        parent.sendall(bad+frame('valid','purple'))
        assert child.consume(history)
        assert history==[{'role':'user','content':'purple'}]
        assert json.loads(parent.recv(1024))=={'id':'valid','accepted':True}
    finally:
        child.close()
        parent.close()


def test_disconnected_parent_does_not_crash_agent():
    parent, child = pair()
    parent.close()
    history = []
    try:
        assert not child.consume(history)
        assert child.closed
        assert history==[]
    finally:
        child.close()


def test_half_closed_sender_can_receive_deferred_receipt():
    parent, child = pair()
    history = []
    try:
        parent.sendall(frame("a", "purple"))
        parent.shutdown(socket.SHUT_WR)
        assert child.consume(history, acknowledge=False)
        assert not child.closed
        child.acknowledge()
        assert json.loads(parent.recv(1024))=={'id':'a','accepted':True}
        assert child.closed
    finally:
        child.close()
        parent.close()


def test_empty_text_frame_is_rejected():
    parent, child = pair()
    history = []
    try:
        parent.sendall(frame('a','')+frame('b','   ')+frame('c','real'))
        assert child.consume(history)
        assert history==[{'role':'user','content':'real'}]
        assert json.loads(parent.recv(1024))=={'id':'c','accepted':True}
    finally:
        child.close()
        parent.close()
