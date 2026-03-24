"""Tests for /restart slash command."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage


def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager"):
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


class TestRestartCommand:

    @pytest.mark.asyncio
    async def test_restart_sends_message_and_calls_execv(self):
        loop, bus = _make_loop()
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/restart")

        with patch("nanobot.agent.loop.os.execv") as mock_execv:
            await loop._handle_restart(msg)
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert "Restarting" in out.content

            await asyncio.sleep(1.5)
            mock_execv.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_intercepted_in_run_loop(self):
        """Verify /restart is handled at the run-loop level, not inside _dispatch."""
        loop, bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/restart")

        with patch.object(loop, "_handle_restart") as mock_handle:
            mock_handle.return_value = None
            await bus.publish_inbound(msg)

            loop._running = True
            run_task = asyncio.create_task(loop.run())
            await asyncio.sleep(0.1)
            loop._running = False
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

            mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_help_includes_restart(self):
        loop, bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/help")

        response = await loop._process_message(msg)

        assert response is not None
        assert "/restart" in response.content
        assert "/ingest" in response.content

    @pytest.mark.asyncio
    async def test_ingest_command_uses_nowcoder_tool(self):
        loop, _ = _make_loop()
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/ingest 14")
        tool = MagicMock()
        tool.execute = AsyncMock(
            return_value='{"written":["demo.md"],"skipped":{"stale":1,"irrelevant":2,"fetch_error":3}}'
        )

        with patch.object(loop.tools, "get", return_value=tool):
            response = await loop._process_message(msg)

        assert response is not None
        assert "Nowcoder ingest finished." in response.content
        assert "updated_within_days: 14" in response.content
        tool.execute.assert_awaited_once_with(
            count_per_query=30,
            max_reports=200,
            fetch_timeout=12,
            dry_run=False,
            updated_within_days=14,
        )
