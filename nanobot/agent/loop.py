"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.digest import ShowDailyDigestTool
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.interview import PrepareMockInterviewTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.nowcoder import CollectNowcoderInterviewsTool
from nanobot.agent.tools.prep import PrepareInterviewPrepTool
from nanobot.agent.tools.recent import ShowRecentReportsTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.review import TriggerReviewTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, WebSearchConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._interviews: dict[str, Any] = {}
        self._processing_lock = asyncio.Lock()
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(CollectNowcoderInterviewsTool())
        self.tools.register(ShowRecentReportsTool())
        self.tools.register(ShowDailyDigestTool())
        self.tools.register(PrepareInterviewPrepTool())
        self.tools.register(PrepareMockInterviewTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        self.tools.register(TriggerReviewTool())

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs,
                model=self.model,
            )

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    tool_hint = self._tool_hint(response.tool_calls)
                    tool_hint = self._strip_think(tool_hint)
                    await on_progress(tool_hint, tool_hint=True)

                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
            elif cmd == "/restart":
                await self._handle_restart(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        self._interviews.pop(msg.session_key, None)
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_restart(self, msg: InboundMessage) -> None:
        """Restart the process in-place via os.execv."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        async def _do_restart():
            await asyncio.sleep(1)
            # Use -m nanobot instead of sys.argv[0] for Windows compatibility
            # (sys.argv[0] may be just "nanobot" without full path on Windows)
            os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

        asyncio.create_task(_do_restart())

    async def _handle_nowcoder_ingest(self, msg: InboundMessage, cmd: str) -> OutboundMessage:
        """Run the built-in Nowcoder ingest flow with simple defaults."""
        parts = cmd.split()
        days = 7
        if len(parts) >= 2:
            try:
                days = max(1, min(int(parts[1]), 90))
            except ValueError:
                pass

        tool = self.tools.get("collect_nowcoder_interviews")
        if tool is None:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Nowcoder ingest tool is not available.",
            )

        result_raw = await tool.execute(
            count_per_query=30,
            max_reports=200,
            fetch_timeout=12,
            dry_run=False,
            updated_within_days=days,
        )
        try:
            result = json.loads(result_raw)
        except Exception:
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result_raw)

        lines = [
            "Nowcoder ingest finished.",
            f"- updated_within_days: {days}",
            f"- written: {len(result.get('written', []))}",
            f"- stale: {result.get('skipped', {}).get('stale', 0)}",
            f"- irrelevant: {result.get('skipped', {}).get('irrelevant', 0)}",
            f"- fetch_error: {result.get('skipped', {}).get('fetch_error', 0)}",
            "- next_step: /recent 7 or /digest 1",
        ]
        written = result.get("written", [])
        if written:
            lines.append(f"- latest: {written[0]}")
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines))

    async def _handle_daily_digest(self, msg: InboundMessage, cmd: str) -> OutboundMessage:
        """Render the local daily digest with simple defaults."""
        parts = cmd.split()
        days = 1
        if len(parts) >= 2:
            try:
                days = max(1, min(int(parts[1]), 30))
            except ValueError:
                pass

        tool = self.tools.get("show_daily_digest")
        if tool is None:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Daily digest tool is not available.",
            )

        result = await tool.execute(days=days)
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

    async def _handle_recent_reports(self, msg: InboundMessage, cmd: str) -> OutboundMessage:
        """Render a concise overview of recently kept materials."""
        parts = cmd.split()
        days = 7
        if len(parts) >= 2:
            try:
                days = max(1, min(int(parts[1]), 30))
            except ValueError:
                pass

        tool = self.tools.get("show_recent_reports")
        if tool is None:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Recent reports tool is not available.",
            )

        result = await tool.execute(days=days, limit=10)
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

    async def _handle_mock_interview(self, msg: InboundMessage, cmd: str) -> OutboundMessage:
        """Start a live interview harness for the current session."""
        topic, resume_path, interview_style = self._parse_interview_command(msg.content.strip())
        try:
            from copilot.app import build_live_interview

            interview = build_live_interview(
                user_id=msg.sender_id,
                topic=topic,
                max_questions=6,
                recent=True,
                candidate_profile_path=resume_path,
                interview_style=interview_style,
            )
        except Exception as exc:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Mock interview is not available: {exc}",
            )

        self._interviews[msg.session_key] = interview
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=interview.open())

    async def _handle_interview_prep(self, msg: InboundMessage, cmd: str) -> OutboundMessage:
        """Build a structured prep pack for the current target interview."""
        topic, resume_path, company, position, target_description = self._parse_prep_command(msg.content.strip())
        tool = self.tools.get("prepare_interview_prep")
        if tool is None:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Interview prep tool is not available.",
            )

        result = await tool.execute(
            topic=topic,
            company=company,
            position=position,
            target_description=target_description,
            candidate_profile_path=resume_path,
            max_questions=8,
            recent=True,
        )
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

    async def _handle_interview_review(self, msg: InboundMessage, session: Session) -> OutboundMessage:
        """Review the current session as an interview dialogue."""
        interview = self._interviews.get(msg.session_key)
        if interview is not None:
            report = interview.review(persist=False)
            result = report.get("summary", "No interview review available.")
            if getattr(interview, "trace_path", None):
                result = f"{result.rstrip()}\n\n- trace: {interview.trace_path}"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

        try:
            from copilot.app import render_interview_review

            result = render_interview_review(session.get_history(max_messages=0), persist=False)
        except Exception as exc:
            result = f"Interview review failed: {exc}"
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

    async def _handle_schedule_digest(self, msg: InboundMessage) -> OutboundMessage:
        """Schedule a daily 9:00 digest for the current chat."""
        tool = self.tools.get("cron")
        if tool is None:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Cron tool is not available.",
            )

        result = await tool.execute(
            action="add",
            message=(
                "Call show_daily_digest with days=1 and send the result back to this chat. "
                "Keep the report concise and structured."
            ),
            cron_expr="0 9 * * *",
            tz="Asia/Shanghai",
        )
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

    @staticmethod
    def _looks_like_greeting(text: str) -> bool:
        normalized = text.strip().lower()
        if normalized in {"hi", "hello", "hey", "/start", "start", "menu", "help"}:
            return True
        return normalized in {"你好", "您好", "在吗", "开始", "菜单", "功能", "怎么用"}

    @staticmethod
    def _is_first_turn(session: Session) -> bool:
        return not session.messages

    @staticmethod
    def _build_help_menu() -> str:
        try:
            from copilot.app import render_assistant_menu

            return render_assistant_menu()
        except Exception:
            return "\n".join(
                [
                    "Interview Copilot",
                    "",
                    "Available actions:",
                    "- /ingest [days]",
                    "- /recent [days]",
                    "- /digest [days]",
                    "- /prep [topic] [--resume path] [--company name] [--position role]",
                    "- /interview [topic] [--resume path]",
                    "- /review",
                    "- answer directly after /interview",
                    "- /help",
                ]
            )

    @staticmethod
    def _parse_interview_command(content: str) -> tuple[str, str, str]:
        command = content.strip()
        payload = command.split(maxsplit=1)[1].strip() if " " in command else ""
        if not payload:
            return "", "", "auto"

        topic = payload
        resume_path = ""
        interview_style = "auto"

        resume_match = re.search(r'--resume\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))', topic)
        if resume_match is not None:
            resume_path = next((group for group in resume_match.groups() if group), "")
            topic = (topic[: resume_match.start()] + topic[resume_match.end() :]).strip()

        style_match = re.search(r'--style\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))', topic)
        if style_match is not None:
            interview_style = next((group for group in style_match.groups() if group), "auto")
            topic = (topic[: style_match.start()] + topic[style_match.end() :]).strip()

        return topic, resume_path, interview_style

    @staticmethod
    def _parse_prep_command(content: str) -> tuple[str, str, str, str, str]:
        command = content.strip()
        payload = command.split(maxsplit=1)[1].strip() if " " in command else ""
        if not payload:
            return "", "", "", "", ""

        topic = payload
        resume_path = ""
        company = ""
        position = ""
        target_description = ""

        for flag, target in (
            ("resume", "resume_path"),
            ("company", "company"),
            ("position", "position"),
            ("target", "target_description"),
        ):
            match = re.search(rf'--{flag}\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))', topic)
            if match is None:
                continue
            value = next((group for group in match.groups() if group), "")
            if target == "resume_path":
                resume_path = value
            elif target == "company":
                company = value
            elif target == "position":
                position = value
            else:
                target_description = value
            topic = (topic[: match.start()] + topic[match.end() :]).strip()

        return topic, resume_path, company, position, target_description

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            snapshot = session.messages[session.last_consolidated:]
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            if snapshot:
                self._schedule_background(self.memory_consolidator.archive_messages(snapshot))
            self._interviews.pop(session.key, None)

            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd.startswith("/ingest"):
            return await self._handle_nowcoder_ingest(msg, cmd)
        if cmd.startswith("/recent"):
            return await self._handle_recent_reports(msg, cmd)
        if cmd.startswith("/digest"):
            return await self._handle_daily_digest(msg, cmd)
        if cmd.startswith("/prep"):
            return await self._handle_interview_prep(msg, cmd)
        if cmd.startswith("/interview"):
            return await self._handle_mock_interview(msg, cmd)
        if cmd == "/review":
            return await self._handle_interview_review(msg, session)
        if cmd == "/schedule_digest":
            return await self._handle_schedule_digest(msg)
        if cmd in {"/menu"}:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=self._build_help_menu(),
            )
        if msg.channel == "dingtalk" and self._is_first_turn(session) and self._looks_like_greeting(msg.content):
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=self._build_help_menu(),
            )
        if cmd == "/help":
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=self._build_help_menu(),
            )
        if key in self._interviews and not self._interviews[key].completed:
            result = self._interviews[key].reply(msg.content)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
