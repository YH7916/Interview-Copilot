from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.digest import ShowDailyDigestTool
from nanobot.agent.tools.interview import PrepareMockInterviewTool
from nanobot.agent.tools.nowcoder import CollectNowcoderInterviewsTool
from nanobot.agent.tools.recent import ShowRecentReportsTool
from nanobot.bus.queue import MessageBus
from nanobot.bus.events import InboundMessage


class _FakeIngestor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "queries": kwargs.get("queries") or ["default"],
            "written": ["demo.md"],
            "skipped": {"irrelevant": 0, "fetch_error": 0},
            "dry_run": kwargs.get("dry_run", False),
        }


@pytest.mark.asyncio
async def test_collect_nowcoder_interviews_tool_uses_custom_query() -> None:
    ingestor = _FakeIngestor()
    tool = CollectNowcoderInterviewsTool(ingestor=ingestor)

    result = await tool.execute(
        query="site:nowcoder.com/discuss Agent 面经",
        count_per_query=3,
        max_reports=2,
        dry_run=True,
        fetch_timeout=8,
        updated_within_days=14,
    )

    assert '"demo.md"' in result
    assert ingestor.calls == [
        {
            "queries": ["site:nowcoder.com/discuss Agent 面经"],
            "count_per_query": 3,
            "max_reports": 2,
            "dry_run": True,
            "rebuild_index": False,
        }
    ]


def test_agent_loop_registers_nowcoder_collection_tool() -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with (
        patch("nanobot.agent.loop.ContextBuilder", return_value=MagicMock()),
        patch("nanobot.agent.loop.SessionManager", return_value=MagicMock()),
        patch("nanobot.agent.loop.SubagentManager", return_value=MagicMock()),
        patch("nanobot.agent.loop.MemoryConsolidator", return_value=MagicMock()),
    ):
        loop = AgentLoop(bus=bus, provider=provider, workspace=Path.cwd(), model="test-model")

    assert loop.tools.has("collect_nowcoder_interviews")
    assert loop.tools.has("show_recent_reports")
    assert loop.tools.has("show_daily_digest")
    assert loop.tools.has("prepare_mock_interview")


@pytest.mark.asyncio
async def test_agent_loop_handles_digest_and_interview_commands() -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    fake_session = SimpleNamespace(messages=[], last_consolidated=0, key="cli:c1")
    fake_sessions = MagicMock()
    fake_sessions.get_or_create.return_value = fake_session

    with (
        patch("nanobot.agent.loop.ContextBuilder", return_value=MagicMock()),
        patch("nanobot.agent.loop.SessionManager", return_value=fake_sessions),
        patch("nanobot.agent.loop.SubagentManager", return_value=MagicMock()),
        patch("nanobot.agent.loop.MemoryConsolidator", return_value=MagicMock()),
    ):
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=Path.cwd(),
            model="test-model",
            cron_service=MagicMock(),
        )

    async def _digest_execute(**kwargs):
        return f"digest:{kwargs['days']}"

    async def _recent_execute(**kwargs):
        return f"recent:{kwargs['days']}:{kwargs['limit']}"

    loop.tools.get("show_recent_reports").execute = _recent_execute  # type: ignore[method-assign]
    loop.tools.get("show_daily_digest").execute = _digest_execute  # type: ignore[method-assign]

    recent = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/recent 5")
    )
    digest = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/digest 3")
    )

    assert recent.content == "recent:5:10"
    assert digest.content == "digest:3"


@pytest.mark.asyncio
async def test_agent_loop_handles_review_command(monkeypatch) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    fake_session = SimpleNamespace(
        messages=[
            {"role": "assistant", "content": "什么是 ReAct？"},
            {"role": "user", "content": "它是推理和行动交替。"},
        ],
        last_consolidated=0,
        key="cli:c1",
        get_history=lambda max_messages=0: [
            {"role": "assistant", "content": "什么是 ReAct？"},
            {"role": "user", "content": "它是推理和行动交替。"},
        ],
    )
    fake_sessions = MagicMock()
    fake_sessions.get_or_create.return_value = fake_session

    with (
        patch("nanobot.agent.loop.ContextBuilder", return_value=MagicMock()),
        patch("nanobot.agent.loop.SessionManager", return_value=fake_sessions),
        patch("nanobot.agent.loop.SubagentManager", return_value=MagicMock()),
        patch("nanobot.agent.loop.MemoryConsolidator", return_value=MagicMock()),
    ):
        loop = AgentLoop(bus=bus, provider=provider, workspace=Path.cwd(), model="test-model")

    with patch("copilot.app.render_interview_review", return_value="# Interview Review\n"):
        response = await loop._process_message(
            InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/review")
        )

    assert response.content == "# Interview Review\n"


@pytest.mark.asyncio
async def test_agent_loop_routes_live_interview_messages() -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    fake_session = SimpleNamespace(messages=[], last_consolidated=0, key="cli:c1")
    fake_sessions = MagicMock()
    fake_sessions.get_or_create.return_value = fake_session

    class _FakeInterview:
        def __init__(self) -> None:
            self.answers = []
            self.completed = False

        def open(self) -> str:
            return "[Q1] 什么是 ReAct？"

        def reply(self, answer: str) -> str:
            self.answers.append(answer)
            return f"next:{answer}"

        def review(self, persist: bool = False) -> dict:
            return {"summary": "# Interview Review\n"}

    interview = _FakeInterview()

    with (
        patch("nanobot.agent.loop.ContextBuilder", return_value=MagicMock()),
        patch("nanobot.agent.loop.SessionManager", return_value=fake_sessions),
        patch("nanobot.agent.loop.SubagentManager", return_value=MagicMock()),
        patch("nanobot.agent.loop.MemoryConsolidator", return_value=MagicMock()),
    ):
        loop = AgentLoop(bus=bus, provider=provider, workspace=Path.cwd(), model="test-model")

    with patch("copilot.app.build_live_interview", return_value=interview):
        start = await loop._process_message(
            InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/interview agent")
        )
        answer = await loop._process_message(
            InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="ReAct 是推理与行动交替。")
        )
        review = await loop._process_message(
            InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/review")
        )

    assert start.content == "[Q1] 什么是 ReAct？"
    assert answer.content == "next:ReAct 是推理与行动交替。"
    assert interview.answers == ["ReAct 是推理与行动交替。"]
    assert review.content == "# Interview Review\n"


@pytest.mark.asyncio
async def test_agent_loop_handles_schedule_digest_command() -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    fake_session = SimpleNamespace(messages=[], last_consolidated=0, key="cli:c1")
    fake_sessions = MagicMock()
    fake_sessions.get_or_create.return_value = fake_session

    with (
        patch("nanobot.agent.loop.ContextBuilder", return_value=MagicMock()),
        patch("nanobot.agent.loop.SessionManager", return_value=fake_sessions),
        patch("nanobot.agent.loop.SubagentManager", return_value=MagicMock()),
        patch("nanobot.agent.loop.MemoryConsolidator", return_value=MagicMock()),
    ):
        loop = AgentLoop(bus=bus, provider=provider, workspace=Path.cwd(), model="test-model")

    async def _cron_execute(**kwargs):
        assert kwargs["action"] == "add"
        assert kwargs["cron_expr"] == "0 9 * * *"
        assert kwargs["tz"] == "Asia/Shanghai"
        assert "show_daily_digest" in kwargs["message"]
        return "Created job 'digest'"

    loop.tools.register(CronTool(MagicMock()))
    loop.tools.get("cron").execute = _cron_execute  # type: ignore[method-assign]

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/schedule_digest")
    )

    assert response.content == "Created job 'digest'"


@pytest.mark.asyncio
async def test_agent_loop_shows_menu_for_first_dingtalk_greeting() -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    fake_session = SimpleNamespace(messages=[], last_consolidated=0, key="dingtalk:c1")
    fake_sessions = MagicMock()
    fake_sessions.get_or_create.return_value = fake_session

    with (
        patch("nanobot.agent.loop.ContextBuilder", return_value=MagicMock()),
        patch("nanobot.agent.loop.SessionManager", return_value=fake_sessions),
        patch("nanobot.agent.loop.SubagentManager", return_value=MagicMock()),
        patch("nanobot.agent.loop.MemoryConsolidator", return_value=MagicMock()),
    ):
        loop = AgentLoop(bus=bus, provider=provider, workspace=Path.cwd(), model="test-model")

    response = await loop._process_message(
        InboundMessage(channel="dingtalk", sender_id="u1", chat_id="c1", content="你好")
    )

    assert "Interview Copilot" in response.content
    assert "/recent [days]" in response.content
    assert "/digest [days]" in response.content
    assert "/review" in response.content
    assert "/schedule_digest" in response.content


@pytest.mark.asyncio
async def test_show_daily_digest_tool_uses_app_facade(monkeypatch) -> None:
    monkeypatch.setattr(
        "copilot.app.render_daily_digest",
        lambda days=1: f"digest:{days}",
    )

    result = await ShowDailyDigestTool().execute(days=3)

    assert result == "digest:3"


@pytest.mark.asyncio
async def test_show_recent_reports_tool_uses_app_facade(monkeypatch) -> None:
    monkeypatch.setattr(
        "copilot.app.render_recent_reports_overview",
        lambda **kwargs: f"recent:{kwargs['days']}:{kwargs['limit']}",
    )

    result = await ShowRecentReportsTool().execute(days=6, limit=4)

    assert result == "recent:6:4"


@pytest.mark.asyncio
async def test_prepare_mock_interview_tool_uses_app_facade(monkeypatch) -> None:
    monkeypatch.setattr(
        "copilot.app.render_mock_interview_plan",
        lambda **kwargs: f"mock:{kwargs['topic']}:{kwargs['max_questions']}:{kwargs['recent']}:{kwargs['interview_style']}",
    )

    result = await PrepareMockInterviewTool().execute(
        topic="Agent",
        max_questions=5,
        recent=False,
        interview_style="no-coding",
    )

    assert result == "mock:Agent:5:False:no-coding"
