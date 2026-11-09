"""
Server-side tests for audio clip import: create_audio_clip's warp overrides
and the create_audio_clips_batch tool. These exercise the real MCP_Server.server
tool functions with the Ableton socket connection mocked, so they run anywhere
— no Ableton, no network.

Run from the repo root:
    uv run pytest -v
    # or, in any env that has the server's deps:
    python -m pytest -v
"""

import asyncio
import inspect

import pytest

import MCP_Server.server as server


def call(tool, *args, **kwargs):
    """Invoke a tool function whichever shape it has (see test_clip_notes.py)."""
    result = tool(*args, **kwargs)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------

class FakeConnection:
    """Stand-in for AbletonConnection. Records every command sent and returns
    a canned response (or raises) so we can test the tool logic in isolation."""

    def __init__(self, response=None, raise_exc=None):
        self.response = {} if response is None else response
        self.raise_exc = raise_exc
        self.sent = []  # list of (command_type, params) tuples

    def send_command(self, command_type, params=None):
        self.sent.append((command_type, params or {}))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


@pytest.fixture
def fake_conn(monkeypatch):
    def _make(response=None, raise_exc=None):
        conn = FakeConnection(response=response, raise_exc=raise_exc)
        monkeypatch.setattr(server, "get_ableton_connection", lambda: conn)
        return conn
    return _make


@pytest.fixture(autouse=True)
def _silence_telemetry(monkeypatch):
    class _Dummy:
        def record_event(self, *a, **k):
            pass
    import MCP_Server.telemetry_decorator as td
    monkeypatch.setattr(td, "get_telemetry", lambda: _Dummy(), raising=False)


@pytest.fixture(autouse=True)
def _no_consent_prompt(monkeypatch):
    import MCP_Server.dataset.consent as consent
    monkeypatch.setattr(consent, "maybe_consent_notice", lambda: "")


@pytest.fixture
def capabilities_available(monkeypatch):
    """Report every capability as present, so the tool body runs rather than
    the missing-script early return. Opt-in (not autouse) so the gating tests
    below can exercise the missing-capability path instead."""
    import MCP_Server.script_handshake as sh
    monkeypatch.setattr(sh, "require_capability", lambda name: None)


# --------------------------------------------------------------------------
# create_audio_clip — warp overrides
# --------------------------------------------------------------------------

def test_create_audio_clip_without_warp_args_omits_them(fake_conn, capabilities_available):
    conn = fake_conn(response={"name": "Clip", "length": 4.0, "warping": True, "warp_mode": 4})
    call(server.create_audio_clip, None, 0, 0, "/abs/path.wav")
    assert conn.sent == [("create_audio_clip",
                          {"track_index": 0, "clip_index": 0, "path": "/abs/path.wav"})]


def test_create_audio_clip_forwards_warp_overrides(fake_conn, capabilities_available):
    conn = fake_conn(response={"name": "Clip", "length": 4.0, "warping": False, "warp_mode": 2})
    call(server.create_audio_clip, None, 0, 0, "/abs/path.wav", warp=False, warp_mode=2)
    assert conn.sent == [("create_audio_clip",
                          {"track_index": 0, "clip_index": 0, "path": "/abs/path.wav",
                           "warp": False, "warp_mode": 2})]


def test_create_audio_clip_rejects_out_of_range_warp_mode_without_calling_ableton(fake_conn, capabilities_available):
    conn = fake_conn()
    out = call(server.create_audio_clip, None, 0, 0, "/abs/path.wav", warp_mode=9)
    assert "warp_mode must be 0-6" in out
    assert conn.sent == []


def test_create_audio_clip_warp_gated_on_missing_capability(fake_conn, monkeypatch):
    import MCP_Server.script_handshake as sh
    monkeypatch.setattr(sh, "require_capability",
                         lambda name: f"missing capability {name}" if name == "create_audio_clip_warp" else None)
    conn = fake_conn()
    out = call(server.create_audio_clip, None, 0, 0, "/abs/path.wav", warp=True)
    assert "create_audio_clip_warp" in out
    assert conn.sent == []  # gated before the connection is ever touched


def test_create_audio_clip_without_warp_args_is_not_capability_gated(fake_conn, monkeypatch):
    """Plain imports (no warp/warp_mode) must keep working against an older
    Remote Script that never heard of the warp capability."""
    import MCP_Server.script_handshake as sh
    monkeypatch.setattr(sh, "require_capability", lambda name: "should not be called")
    conn = fake_conn(response={"name": "Clip", "length": 4.0})
    out = call(server.create_audio_clip, None, 0, 0, "/abs/path.wav")
    assert "Created audio clip" in out
    assert len(conn.sent) == 1


def test_create_audio_clip_connection_error_is_reported(fake_conn, capabilities_available):
    fake_conn(raise_exc=Exception("boom"))
    out = call(server.create_audio_clip, None, 0, 0, "/abs/path.wav")
    assert out.startswith("Error creating audio clip:")
    assert "boom" in out


# --------------------------------------------------------------------------
# create_audio_clips_batch
# --------------------------------------------------------------------------

def test_batch_rejects_empty_list(fake_conn, capabilities_available):
    conn = fake_conn()
    out = call(server.create_audio_clips_batch, None, [])
    assert "empty" in out
    assert conn.sent == []


def test_batch_rejects_over_the_limit(fake_conn, capabilities_available):
    conn = fake_conn()
    clips = [{"track_index": 0, "clip_index": i, "path": "/a.wav"}
             for i in range(server.MAX_BATCH_CLIPS + 1)]
    out = call(server.create_audio_clips_batch, None, clips)
    assert str(server.MAX_BATCH_CLIPS) in out
    assert "exceeds" in out
    assert conn.sent == []


def test_batch_rejects_clip_missing_required_field(fake_conn, capabilities_available):
    conn = fake_conn()
    out = call(server.create_audio_clips_batch, None, [{"track_index": 0, "path": "/a.wav"}])
    assert "index 0" in out
    assert "required field" in out
    assert conn.sent == []


def test_batch_rejects_invalid_warp_mode_before_sending(fake_conn, capabilities_available):
    conn = fake_conn()
    out = call(server.create_audio_clips_batch, None,
               [{"track_index": 0, "clip_index": 0, "path": "/a.wav", "warp_mode": 12}])
    assert "index 0" in out
    assert "warp_mode" in out
    assert conn.sent == []


def test_batch_gated_on_missing_batch_capability(fake_conn, monkeypatch):
    import MCP_Server.script_handshake as sh
    monkeypatch.setattr(sh, "require_capability",
                         lambda name: "missing" if name == "create_audio_clips_batch" else None)
    conn = fake_conn()
    out = call(server.create_audio_clips_batch, None,
               [{"track_index": 0, "clip_index": 0, "path": "/a.wav"}])
    assert "missing" in out
    assert conn.sent == []


def test_batch_gated_on_missing_warp_capability_only_when_warp_used(fake_conn, monkeypatch):
    import MCP_Server.script_handshake as sh
    monkeypatch.setattr(sh, "require_capability",
                         lambda name: "missing warp" if name == "create_audio_clip_warp" else None)
    conn = fake_conn()

    # No warp fields used -> the warp capability is never consulted.
    conn.response = {"results": [{"index": 0, "status": "success", "track_index": 0,
                                  "clip_index": 0, "name": "Clip", "length": 4.0}]}
    out = call(server.create_audio_clips_batch, None,
               [{"track_index": 0, "clip_index": 0, "path": "/a.wav"}])
    assert "Imported 1/1" in out

    # A warp field anywhere in the batch -> gated.
    out = call(server.create_audio_clips_batch, None,
               [{"track_index": 0, "clip_index": 0, "path": "/a.wav"},
                {"track_index": 0, "clip_index": 1, "path": "/b.wav", "warp": True}])
    assert "missing warp" in out


def test_batch_forwards_clips_list_verbatim(fake_conn, capabilities_available):
    clips = [
        {"track_index": 0, "clip_index": 0, "path": "/a.wav"},
        {"track_index": 1, "clip_index": 2, "path": "/b.wav", "warp": True, "warp_mode": 4},
    ]
    conn = fake_conn(response={"results": []})
    call(server.create_audio_clips_batch, None, clips)
    assert conn.sent == [("create_audio_clips_batch", {"clips": clips})]


def test_batch_summarises_mixed_results(fake_conn, capabilities_available):
    fake_conn(response={"results": [
        {"index": 0, "track_index": 0, "clip_index": 0, "status": "success",
         "name": "Kick", "length": 4.0, "warping": True},
        {"index": 1, "track_index": 0, "clip_index": 1, "status": "error",
         "message": "Clip slot already has a clip"},
    ]})
    out = call(server.create_audio_clips_batch, None, [
        {"track_index": 0, "clip_index": 0, "path": "/kick.wav"},
        {"track_index": 0, "clip_index": 1, "path": "/snare.wav"},
    ])
    assert "Imported 1/2 clips (1 failed)" in out
    assert "Kick" in out
    assert "FAILED - Clip slot already has a clip" in out


def test_batch_connection_error_is_reported(fake_conn, capabilities_available):
    fake_conn(raise_exc=Exception("boom"))
    out = call(server.create_audio_clips_batch, None,
               [{"track_index": 0, "clip_index": 0, "path": "/a.wav"}])
    assert out.startswith("Error creating audio clips batch:")
    assert "boom" in out


# --------------------------------------------------------------------------
# create_scene
# --------------------------------------------------------------------------

def test_create_scene_sends_index(fake_conn, capabilities_available):
    conn = fake_conn(response={"index": 2, "name": "Scene"})
    out = call(server.create_scene, None)
    assert conn.sent == [("create_scene", {"index": -1})]
    assert "Created new scene 'Scene' at index 2" in out


def test_create_scene_forwards_explicit_index(fake_conn, capabilities_available):
    conn = fake_conn(response={"index": 3, "name": "Scene"})
    call(server.create_scene, None, index=3)
    assert conn.sent == [("create_scene", {"index": 3})]


def test_create_scene_gated_on_missing_capability(fake_conn, monkeypatch):
    import MCP_Server.script_handshake as sh
    monkeypatch.setattr(sh, "require_capability",
                         lambda name: f"missing capability {name}" if name == "create_scene" else None)
    conn = fake_conn()
    out = call(server.create_scene, None)
    assert "create_scene" in out
    assert conn.sent == []  # gated before the connection is ever touched


def test_create_scene_connection_error_is_reported(fake_conn, capabilities_available):
    fake_conn(raise_exc=Exception("boom"))
    out = call(server.create_scene, None)
    assert out.startswith("Error creating scene:")
    assert "boom" in out
