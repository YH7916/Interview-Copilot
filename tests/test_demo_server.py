from copilot.demo_server import DemoRuntime, STATIC_DIR


class _FakeAgentLoop:
    async def process_direct(self, content, session_key="cli:direct", channel="cli", chat_id="direct", on_progress=None):
        del session_key, channel, chat_id, on_progress
        return f"echo:{content}"


def test_demo_runtime_handles_message():
    runtime = DemoRuntime(agent_loop=_FakeAgentLoop(), model="demo-model")

    result = runtime.handle_message("/help")

    assert result == "echo:/help"


def test_demo_static_assets_exist():
    assert (STATIC_DIR / "index.html").exists()
    assert (STATIC_DIR / "styles.css").exists()
    assert (STATIC_DIR / "app.js").exists()
