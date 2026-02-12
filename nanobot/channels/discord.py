"""Discord channel implementation using discord.py."""

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import discord
from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.progress import ProgressEvent, ProgressKind
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import DiscordConfig

if TYPE_CHECKING:
    from discord import Message as DiscordMessage


class DiscordChannel(BaseChannel):
    """
    Discord channel using discord.py with WebSocket gateway connection.

    Responds to:
    - @mentions of the bot
    - Messages containing the configured trigger word (default: "nano")
    - Replies to the bot's messages
    """

    name = "discord"

    # Discord embed color (Discord blurple)
    EMBED_COLOR = 0x5865F2
    # Maximum embed description length
    EMBED_MAX_LENGTH = 4096
    # Maximum message length
    MESSAGE_MAX_LENGTH = 2000

    def __init__(self, config: DiscordConfig, bus: MessageBus, groq_api_key: str = ""):
        super().__init__(config, bus)
        self.config: DiscordConfig = config
        self.groq_api_key = groq_api_key
        self._client: discord.Client | None = None
        self._guild: discord.Guild | None = None
        # Track messages being processed for reaction management
        self._processing_messages: dict[int, tuple[int, "DiscordMessage"]] = {}
        # Track typing indicator tasks
        self._typing_tasks: dict[int, asyncio.Task] = {}
        # Track sent messages for edit support (chat_id -> last message)
        self._sent_messages: dict[str, discord.Message] = {}
        # Track progress status messages per chat
        self._progress_messages: dict[str, discord.Message] = {}
        # Track active threads for long-running tasks
        self._active_threads: dict[str, discord.Thread] = {}

    async def start(self) -> None:
        """Start the Discord bot with WebSocket gateway connection."""
        if not self.config.token:
            logger.error("Discord bot token not configured")
            return

        if not self.config.guild_id:
            logger.error("Discord guild_id not configured")
            return

        self._running = True

        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True  # Privileged intent - required for reading message text
        intents.guilds = True
        intents.guild_messages = True
        intents.reactions = True

        # Create client
        self._client = discord.Client(intents=intents)

        # Register event handlers
        @self._client.event
        async def on_ready() -> None:
            if self._client and self._client.user:
                logger.info(f"Discord bot {self._client.user.name} connected")

                # Get the configured guild
                self._guild = self._client.get_guild(int(self.config.guild_id))
                if self._guild:
                    logger.info(f"Connected to guild: {self._guild.name}")

                    # Validate default notification channel
                    if self.config.default_channel_id:
                        default_ch = self._guild.get_channel(int(self.config.default_channel_id))
                        if default_ch:
                            logger.info(
                                f"Validated default notification "
                                f"channel: #{default_ch.name} "
                                f"({self.config.default_channel_id})"
                            )
                        else:
                            logger.warning(
                                f"default_channel_id "
                                f"{self.config.default_channel_id}"
                                f" not found in guild "
                                f"'{self._guild.name}'. Proactive "
                                "messages will fail until this "
                                "is fixed."
                            )
                    else:
                        logger.warning(
                            "No default_channel_id configured "
                            "for Discord. Cron delivery and "
                            "heartbeat notifications require "
                            "this setting."
                        )
                else:
                    logger.warning(f"Could not find guild with ID: {self.config.guild_id}")

        @self._client.event
        async def on_message(message: "DiscordMessage") -> None:
            await self._on_message(message)

        logger.info("Starting Discord bot...")

        # Use client.start() for async integration (not client.run() which is blocking)
        try:
            await self._client.start(self.config.token)
        except discord.LoginFailure:
            logger.error("Discord login failed - check your bot token")
            self._running = False
        except Exception as e:
            logger.error(f"Discord connection error: {e}")
            self._running = False

    async def stop(self) -> None:
        """Stop the Discord bot."""
        self._running = False

        # Cancel all typing tasks
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()

        if self._client:
            logger.info("Stopping Discord bot...")
            await self._client.close()
            self._client = None
            self._guild = None

    async def _start_typing(self, message_id: int, channel: discord.TextChannel) -> None:
        """Start a background task that maintains typing indicator."""

        async def typing_loop() -> None:
            try:
                while True:
                    await channel.typing()
                    await asyncio.sleep(8)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Typing indicator stopped: {e}")

        task = asyncio.create_task(typing_loop())
        self._typing_tasks[message_id] = task

    async def _stop_typing(self, message_id: int) -> None:
        """Stop the typing indicator task."""
        if message_id in self._typing_tasks:
            task = self._typing_tasks.pop(message_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Discord."""
        if not self._client or not self._client.is_ready():
            logger.warning("Discord bot not running")
            return

        # Get original message for reply (don't mark complete yet - do that after sending)
        # Note: original_msg_id may be string after JSON serialization through message bus
        raw_msg_id = msg.metadata.get("original_message_id")
        original_msg_id: int | None = int(raw_msg_id) if raw_msg_id else None
        original_message: "DiscordMessage | None" = None

        logger.debug(f"send() called - raw_msg_id={raw_msg_id}, original_msg_id={original_msg_id}")
        logger.debug(f"_processing_messages keys: {list(self._processing_messages.keys())}")

        if original_msg_id and original_msg_id in self._processing_messages:
            _, original_message = self._processing_messages[original_msg_id]
            logger.debug(f"Found original message for reply: {original_message.id}")
        else:
            logger.debug("No original message found for reply")

        # Determine target channel
        channel_id = msg.chat_id
        if not channel_id or channel_id == "default":
            # Use default channel for proactive messages
            channel_id = self.config.default_channel_id

        if not channel_id:
            logger.error("No channel_id specified and no default_channel_id configured")
            return

        try:
            channel = self._client.get_channel(int(channel_id))
            if not channel:
                channel = await self._client.fetch_channel(int(channel_id))

            if not isinstance(channel, discord.TextChannel):
                logger.error(f"Channel {channel_id} is not a text channel")
                return

            # Delegate to edit() if this is an edit request
            if msg.edit_message_id:
                await self.edit(msg)
                return

            # Clean up progress message for this chat
            progress_key = f"{msg.channel}:{msg.chat_id}"
            progress_msg = self._progress_messages.pop(progress_key, None)
            if progress_msg:
                try:
                    await progress_msg.delete()
                except (discord.NotFound, discord.HTTPException):
                    pass

            # Send as reply to original message (if we have it)
            content = msg.content
            view = None
            if msg.components:
                view = self._build_discord_view(msg.components, msg.chat_id)
            await self._send_message(channel, content, reply_to=original_message, view=view)

            # Mark complete AFTER message is sent (updates reaction and stops typing)
            if original_msg_id:
                await self._mark_complete(original_msg_id)

        except discord.Forbidden:
            logger.error(f"Permission denied sending to channel {channel_id}")
        except discord.NotFound:
            logger.error(f"Channel {channel_id} not found")
        except ValueError:
            logger.error(f"Invalid channel_id: {channel_id}")
        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")

    async def on_progress(self, event: ProgressEvent) -> None:
        """Display progress events as an updating status embed."""
        if not self._client or not self._client.is_ready():
            return

        key = f"{event.channel}:{event.chat_id}"

        try:
            channel = self._client.get_channel(int(event.chat_id))
            if not isinstance(channel, discord.TextChannel):
                return

            # Route long-running tasks to a thread
            threshold = getattr(self.config, "auto_thread_threshold", 3)
            if event.iteration > threshold:
                thread_key = f"thread:{key}"
                if thread_key not in self._active_threads:
                    try:
                        thread = await self._create_thread(
                            channel, f"Processing ({event.chat_id[:8]}...)"
                        )
                        self._active_threads[thread_key] = thread
                    except Exception as e:
                        logger.debug(f"Thread creation failed: {e}")
                if thread_key in self._active_threads:
                    channel = self._active_threads[thread_key]

            # Build status line from event
            if event.kind == ProgressKind.THINKING:
                status = f"Thinking... (step {event.iteration}/{event.total_iterations})"
            elif event.kind == ProgressKind.TOOL_START:
                status = f"Running `{event.tool_name}`..."
            elif event.kind == ProgressKind.TOOL_COMPLETE:
                status = f"`{event.tool_name}` {event.detail}"
            elif event.kind == ProgressKind.STREAMING:
                if event.detail:
                    await self._streaming_embed_edit(
                        channel, key, event.detail, self.EMBED_MAX_LENGTH
                    )
                    return
                status = "Generating response..."
            elif event.kind == ProgressKind.ERROR:
                status = f"Error: {event.detail}"
            else:
                status = event.detail or event.kind.value

            embed = discord.Embed(
                description=status,
                color=0x5865F2,
            )
            embed.set_footer(text="Processing")

            if key in self._progress_messages:
                try:
                    await self._progress_messages[key].edit(embed=embed)
                except (discord.NotFound, discord.HTTPException):
                    self._progress_messages.pop(key, None)
            else:
                msg = await channel.send(embed=embed)
                self._progress_messages[key] = msg

        except Exception as e:
            logger.debug(f"Progress display failed: {e}")

    async def _streaming_embed_edit(
        self,
        channel: discord.abc.Messageable,
        key: str,
        content: str,
        max_length: int,
    ) -> None:
        if len(content) > max_length:
            content = content[: max_length - 3] + "..."

        embed = discord.Embed(description=content, color=0x5865F2)
        embed.set_footer(text="Generating...")

        if key in self._progress_messages:
            try:
                await self._progress_messages[key].edit(embed=embed)
            except (discord.NotFound, discord.HTTPException):
                self._progress_messages.pop(key, None)
        else:
            msg = await channel.send(embed=embed)
            self._progress_messages[key] = msg

    async def edit(self, msg: OutboundMessage) -> None:
        """Edit a previously sent message by edit_message_id."""
        if not self._client or not self._client.is_ready():
            return

        if not msg.edit_message_id:
            return

        try:
            channel = self._client.get_channel(int(msg.chat_id))
            if not isinstance(channel, discord.TextChannel):
                return

            message = await channel.fetch_message(int(msg.edit_message_id))
            if len(msg.content) <= self.MESSAGE_MAX_LENGTH:
                await message.edit(content=msg.content)
            else:
                # Switch to embed for long content
                embed = self._create_embed(msg.content)
                await message.edit(content=None, embed=embed)

        except discord.NotFound:
            logger.debug(f"Message {msg.edit_message_id} not found for edit")
        except Exception as e:
            logger.warning(f"Failed to edit Discord message: {e}")

    def _build_discord_view(self, components: list[dict], chat_id: str) -> discord.ui.View:
        """Build a Discord UI View from component definitions."""
        view = discord.ui.View(timeout=300)

        for comp in components:
            if comp.get("type") == "button":
                button = discord.ui.Button(
                    label=comp.get("label", ""),
                    custom_id=comp.get("callback_data", ""),
                    style=discord.ButtonStyle.primary,
                )

                callback_data = comp.get("callback_data", "")
                bus = self.bus
                channel_name = self.name

                async def button_callback(
                    interaction: discord.Interaction,
                    data: str = callback_data,
                ) -> None:
                    await interaction.response.defer()
                    sender_id = str(interaction.user.id)
                    if interaction.user.name:
                        sender_id = f"{sender_id}|{interaction.user.name}"
                    await bus.publish_inbound(
                        InboundMessage(
                            channel=channel_name,
                            sender_id=sender_id,
                            chat_id=chat_id,
                            content=data,
                        )
                    )

                button.callback = button_callback
                view.add_item(button)

            elif comp.get("type") == "select":
                options = [
                    discord.SelectOption(
                        label=opt.get("label", ""),
                        value=opt.get("value", ""),
                    )
                    for opt in comp.get("options", [])
                ]
                select = discord.ui.Select(
                    placeholder=comp.get("placeholder", "Choose..."),
                    options=options or [discord.SelectOption(label="None", value="none")],
                )

                bus = self.bus
                channel_name = self.name

                async def select_callback(
                    interaction: discord.Interaction,
                    s: discord.ui.Select = select,
                ) -> None:
                    await interaction.response.defer()
                    sender_id = str(interaction.user.id)
                    if interaction.user.name:
                        sender_id = f"{sender_id}|{interaction.user.name}"
                    value = s.values[0] if s.values else ""
                    await bus.publish_inbound(
                        InboundMessage(
                            channel=channel_name,
                            sender_id=sender_id,
                            chat_id=chat_id,
                            content=value,
                        )
                    )

                select.callback = select_callback
                view.add_item(select)

        return view

    async def _create_thread(
        self,
        channel: discord.TextChannel,
        name: str,
        message: discord.Message | None = None,
    ) -> discord.Thread:
        """Create a thread for complex task updates."""
        thread = await channel.create_thread(
            name=name,
            message=message,
            auto_archive_duration=60,
        )
        return thread

    async def _on_message(self, message: "DiscordMessage") -> None:
        """Handle incoming Discord messages."""
        # Skip messages from bots (including self)
        if message.author.bot:
            return

        # Skip if no client or guild
        if not self._client or not self._client.user:
            return

        # Check if message is from configured guild
        if not message.guild or str(message.guild.id) != self.config.guild_id:
            return

        # Check if we should respond (mention, trigger word, or reply to bot)
        if not await self._should_respond(message):
            return

        # Build sender_id (format: user_id|username to match Telegram pattern)
        sender_id = str(message.author.id)
        if message.author.name:
            sender_id = f"{sender_id}|{message.author.name}"

        # Check permissions
        if not self.is_allowed(sender_id):
            logger.debug(f"Message from {sender_id} not allowed")
            return

        # Fetch channel history context if configured
        channel_context = ""
        if self.config.context_messages > 0:
            channel_context = await self._fetch_channel_context(message)

        # Add processing reaction
        try:
            await message.add_reaction(self.config.emoji_processing)
            self._processing_messages[message.id] = (message.channel.id, message)
        except discord.Forbidden:
            logger.warning("Cannot add reactions - missing permissions")
        except Exception as e:
            logger.warning(f"Failed to add processing reaction: {e}")

        # Start typing indicator
        if isinstance(message.channel, discord.TextChannel):
            await self._start_typing(message.id, message.channel)

        # Extract reply context (the message being replied to)
        reply_context = await self._fetch_reply_context(message)

        # Extract and clean content
        content = self._clean_message_content(message)

        # Handle attachments
        media_paths: list[str] = []
        content_parts = [content] if content else []

        for attachment in message.attachments:
            try:
                media_path = await self._download_attachment(attachment)
                if media_path:
                    media_paths.append(media_path)

                    # Handle audio transcription
                    if self._is_audio_file(attachment.filename):
                        from nanobot.providers.transcription import GroqTranscriptionProvider

                        transcriber = GroqTranscriptionProvider(api_key=self.groq_api_key)
                        transcription = await transcriber.transcribe(Path(media_path))
                        if transcription:
                            logger.info(f"Transcribed audio: {transcription[:50]}...")
                            content_parts.append(f"[transcription: {transcription}]")
                        else:
                            content_parts.append(f"[audio: {media_path}]")
                    else:
                        content_parts.append(f"[attachment: {media_path}]")
            except Exception as e:
                logger.error(f"Failed to download attachment: {e}")
                content_parts.append(f"[attachment: download failed - {attachment.filename}]")

        final_content = "\n".join(content_parts) if content_parts else "[empty message]"

        # Build mention context (sender info + mention mappings)
        mention_ctx = self._build_mention_context(message, final_content)
        final_content = f"{mention_ctx}\n{final_content}"

        logger.debug(f"Discord message from {sender_id}: {final_content[:50]}...")

        # Forward to message bus
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(message.channel.id),
            content=final_content,
            media=media_paths,
            metadata={
                "message_id": message.id,
                "original_message_id": message.id,
                "user_id": message.author.id,
                "username": message.author.name,
                "display_name": message.author.display_name,
                "channel_name": getattr(message.channel, "name", "unknown"),
                "guild_id": message.guild.id if message.guild else None,
                "guild_name": message.guild.name if message.guild else None,
                "channel_context": channel_context,
                "reply_context": reply_context,
            },
        )

    async def _should_respond(self, message: "DiscordMessage") -> bool:
        """Check if the bot should respond to this message."""
        if not self._client or not self._client.user:
            return False

        # Check for bot mention
        if self._client.user.mentioned_in(message):
            return True

        # Check for trigger word (case-insensitive)
        if self.config.trigger_word:
            pattern = re.compile(rf"\b{re.escape(self.config.trigger_word)}\b", re.IGNORECASE)
            if pattern.search(message.content):
                return True

        # Check if this is a reply to the bot's message
        if message.reference and message.reference.message_id:
            # Check cached reference first (avoids API call)
            if message.reference.resolved:
                if hasattr(message.reference.resolved, "author"):
                    if message.reference.resolved.author.id == self._client.user.id:
                        return True
            else:
                # Fetch the referenced message to check author
                try:
                    referenced_msg = await message.channel.fetch_message(
                        message.reference.message_id
                    )
                    if referenced_msg.author.id == self._client.user.id:
                        return True
                except discord.NotFound:
                    logger.debug(f"Referenced message {message.reference.message_id} not found")
                except discord.Forbidden:
                    logger.debug("No permission to fetch referenced message")
                except Exception as e:
                    logger.warning(f"Failed to fetch referenced message: {e}")

        return False

    async def _fetch_channel_context(self, message: "DiscordMessage") -> str:
        """Fetch recent channel messages as context.

        The Discord API ``limit`` counts all messages including bot messages.
        Since bot messages are filtered out (they exist in session history),
        we over-fetch by 3x and stop once we have collected the desired
        number of non-bot messages.
        """
        try:
            target = self.config.context_messages
            history_messages: list[str] = []
            # Over-fetch to compensate for bot messages being filtered out
            fetch_limit = target * 3
            async for msg in message.channel.history(limit=fetch_limit, before=message):
                # Skip bot messages (we have our own session history)
                if msg.author.bot:
                    continue

                # Format: "username: message content"
                author = msg.author.display_name or msg.author.name
                content = msg.content[:500]  # Truncate very long messages
                if msg.guild:
                    content = self._resolve_mentions(content, msg.guild)
                history_messages.append(f"{author}: {content}")

                if len(history_messages) >= target:
                    break

            if not history_messages:
                return ""

            # Reverse to chronological order (oldest first)
            history_messages.reverse()

            return "\n".join(history_messages)
        except Exception as e:
            logger.warning(f"Failed to fetch channel history: {e}")
            return ""

    async def _fetch_reply_context(self, message: "DiscordMessage") -> str:
        """Fetch the content of the message being replied to, if any.

        When a user replies to a specific message, this extracts the
        referenced message so the agent knows what is being referred to.
        """
        if not message.reference or not message.reference.message_id:
            return ""

        try:
            # Use cached resolved reference if available
            ref_msg = None
            if message.reference.resolved and hasattr(message.reference.resolved, "content"):
                ref_msg = message.reference.resolved
            else:
                ref_msg = await message.channel.fetch_message(message.reference.message_id)

            if not ref_msg:
                return ""

            author = ref_msg.author.display_name or ref_msg.author.name
            content = ref_msg.content[:500]
            if ref_msg.guild:
                content = self._resolve_mentions(content, ref_msg.guild)

            return f"{author}: {content}"
        except discord.NotFound:
            logger.debug(f"Referenced message {message.reference.message_id} not found")
            return ""
        except discord.Forbidden:
            logger.debug("No permission to fetch referenced message")
            return ""
        except Exception as e:
            logger.warning(f"Failed to fetch reply context: {e}")
            return ""

    def _clean_message_content(self, message: "DiscordMessage") -> str:
        """Remove bot mention and trigger word from message content."""
        content = message.content

        # Remove bot mention
        if self._client and self._client.user:
            mention_patterns = [
                f"<@{self._client.user.id}>",
                f"<@!{self._client.user.id}>",
            ]
            for pattern in mention_patterns:
                content = content.replace(pattern, "")

        # Remove trigger word (keeping the rest of the sentence)
        if self.config.trigger_word:
            pattern = re.compile(
                rf"\b{re.escape(self.config.trigger_word)}\b[,:]?\s*", re.IGNORECASE
            )
            content = pattern.sub("", content)

        return content.strip()

    def _resolve_mentions(self, content: str, guild: discord.Guild) -> str:
        """Resolve <@user_id> mentions to @DisplayName for readability."""

        def replace_mention(match: re.Match) -> str:
            user_id = int(match.group(1))
            member = guild.get_member(user_id)
            if member:
                return f"@{member.display_name}"
            return match.group(0)

        return re.sub(r"<@!?(\d+)>", replace_mention, content)

    def _build_mention_context(self, message: "DiscordMessage", content: str) -> str:
        """Build a context header with sender info and mention mappings."""
        parts = []

        # Sender identity with mention syntax
        display = message.author.display_name or message.author.name
        parts.append(f"[From: {display} <@{message.author.id}>]")

        # Resolve any user mentions in the content to a mapping
        if message.mentions and message.guild:
            mappings = []
            for user in message.mentions:
                # Skip bot's own mention (already stripped from content)
                if self._client and self._client.user and user.id == self._client.user.id:
                    continue
                member = message.guild.get_member(user.id)
                name = member.display_name if member else user.display_name
                mappings.append(f"<@{user.id}> = {name}")
            if mappings:
                parts.append(f"[Mentions: {', '.join(mappings)}]")

        return "\n".join(parts)

    def _create_embed(self, content: str) -> discord.Embed:
        """Create a formatted Discord embed for the response."""
        embed = discord.Embed(
            description=content[: self.EMBED_MAX_LENGTH],
            color=self.EMBED_COLOR,
            timestamp=datetime.now(timezone.utc),
        )
        embed.set_footer(text="Nanobot")
        return embed

    async def _send_message(
        self,
        channel: discord.TextChannel,
        content: str,
        reply_to: "DiscordMessage | None" = None,
        view: discord.ui.View | None = None,
    ) -> None:
        """Send a message, splitting if it exceeds Discord's 2000 char limit."""
        if len(content) <= self.MESSAGE_MAX_LENGTH:
            if reply_to:
                await channel.send(content, reference=reply_to, view=view)
            else:
                await channel.send(content, view=view)
        else:
            chunks = self._split_content(content, self.MESSAGE_MAX_LENGTH)
            for i, chunk in enumerate(chunks):
                # Only reply with the first chunk
                if i == 0 and reply_to:
                    await channel.send(chunk, reference=reply_to)
                else:
                    await channel.send(chunk)

    async def _send_long_message(self, channel: discord.TextChannel, content: str) -> None:
        """Send a long message by splitting into multiple parts."""
        # Try to split at paragraph or sentence boundaries
        chunks = self._split_content(content, self.EMBED_MAX_LENGTH)

        for i, chunk in enumerate(chunks):
            embed = discord.Embed(
                description=chunk,
                color=self.EMBED_COLOR,
            )
            if i == len(chunks) - 1:
                # Only add footer and timestamp to last embed
                embed.timestamp = datetime.now(timezone.utc)
                embed.set_footer(text="Nanobot")

            await channel.send(embed=embed)

    def _split_content(self, content: str, max_length: int) -> list[str]:
        """Split content into chunks at natural boundaries."""
        if len(content) <= max_length:
            return [content]

        chunks = []
        remaining = content

        while remaining:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                break

            # Try to find a good split point
            split_point = max_length

            # Look for paragraph break
            newline_pos = remaining.rfind("\n\n", 0, max_length)
            if newline_pos > max_length // 2:
                split_point = newline_pos + 2
            else:
                # Look for single newline
                newline_pos = remaining.rfind("\n", 0, max_length)
                if newline_pos > max_length // 2:
                    split_point = newline_pos + 1
                else:
                    # Look for sentence end
                    for sep in [". ", "! ", "? "]:
                        pos = remaining.rfind(sep, 0, max_length)
                        if pos > max_length // 2:
                            split_point = pos + 2
                            break

            chunks.append(remaining[:split_point].rstrip())
            remaining = remaining[split_point:].lstrip()

        return chunks

    async def _mark_complete(self, message_id: int) -> None:
        """Mark a message as complete by updating reactions."""
        # Stop typing indicator first
        await self._stop_typing(message_id)

        if message_id not in self._processing_messages:
            return

        _, message = self._processing_messages.pop(message_id)

        try:
            # Add complete reaction first (before removing processing, to avoid visual jump)
            await message.add_reaction(self.config.emoji_complete)
            # Then remove processing reaction
            if self._client and self._client.user:
                await message.remove_reaction(self.config.emoji_processing, self._client.user)
        except discord.Forbidden:
            logger.debug("Cannot modify reactions - missing permissions")
        except discord.NotFound:
            logger.debug("Message not found for reaction update")
        except Exception as e:
            logger.warning(f"Failed to update reactions: {e}")

    async def _mark_error(self, message_id: int) -> None:
        """Mark a message as failed by updating reactions."""
        # Stop typing indicator first
        await self._stop_typing(message_id)

        if message_id not in self._processing_messages:
            return

        _, message = self._processing_messages.pop(message_id)

        try:
            # Add error reaction first (before removing processing, to avoid visual jump)
            await message.add_reaction(self.config.emoji_error)
            # Then remove processing reaction
            if self._client and self._client.user:
                await message.remove_reaction(self.config.emoji_processing, self._client.user)
        except discord.Forbidden:
            logger.debug("Cannot modify reactions - missing permissions")
        except discord.NotFound:
            logger.debug("Message not found for reaction update")
        except Exception as e:
            logger.warning(f"Failed to update reactions: {e}")

    async def _download_attachment(self, attachment: discord.Attachment) -> str | None:
        """Download a Discord attachment to local storage."""
        try:
            # Create media directory
            media_dir = Path.home() / ".nanobot" / "media"
            media_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            ext = Path(attachment.filename).suffix or ""
            file_path = media_dir / f"discord_{attachment.id}{ext}"

            # Download
            await attachment.save(file_path)
            logger.debug(f"Downloaded attachment to {file_path}")

            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to download attachment {attachment.filename}: {e}")
            return None

    def _is_audio_file(self, filename: str) -> bool:
        """Check if a file is an audio file based on extension."""
        audio_extensions = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus"}
        ext = Path(filename).suffix.lower()
        return ext in audio_extensions

    def get_guild(self) -> discord.Guild | None:
        """Get the connected Discord guild."""
        return self._guild

    def get_client(self) -> discord.Client | None:
        """Get the Discord client."""
        return self._client

    async def update_default_channel(self, channel_id: str) -> None:
        """Update the default channel ID at runtime."""
        self.config.default_channel_id = channel_id
        logger.info(f"Updated default Discord channel to: {channel_id}")
