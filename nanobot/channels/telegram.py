"""Telegram channel implementation using python-telegram-bot."""

import asyncio
import re

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.progress import ProgressEvent, ProgressKind
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import TelegramConfig


def _markdown_to_telegram_html(text: str) -> str:
    """
    Convert markdown to Telegram-safe HTML.
    """
    if not text:
        return ""

    # 1. Extract and protect code blocks (preserve content from other processing)
    code_blocks: list[str] = []

    def save_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r"```[\w]*\n?([\s\S]*?)```", save_code_block, text)

    # 2. Extract and protect inline code
    inline_codes: list[str] = []

    def save_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", save_inline_code, text)

    # 3. Headers # Title -> just the title text
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)

    # 4. Blockquotes > text -> just the text (before HTML escaping)
    text = re.sub(r"^>\s*(.*)$", r"\1", text, flags=re.MULTILINE)

    # 5. Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 6. Links [text](url) - must be before bold/italic to handle nested cases
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    # 7. Bold **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # 8. Italic _text_ (avoid matching inside words like some_var_name)
    text = re.sub(r"(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])", r"<i>\1</i>", text)

    # 9. Strikethrough ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # 10. Bullet lists - item -> • item
    text = re.sub(r"^[-*]\s+", "• ", text, flags=re.MULTILINE)

    # 11. Restore inline code with HTML tags
    for i, code in enumerate(inline_codes):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")

    # 12. Restore code blocks with HTML tags
    for i, code in enumerate(code_blocks):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")

    return text


class TelegramChannel(BaseChannel):
    """
    Telegram channel using long polling.

    Simple and reliable - no webhook/public IP needed.
    """

    name = "telegram"

    def __init__(self, config: TelegramConfig, bus: MessageBus, groq_api_key: str = ""):
        super().__init__(config, bus)
        self.config: TelegramConfig = config
        self.groq_api_key = groq_api_key
        self._app: Application | None = None
        self._chat_ids: dict[str, int] = {}  # Map sender_id to chat_id for replies
        # Track progress status messages per chat for edit-in-place
        self._progress_messages: dict[str, int] = {}  # chat_id -> message_id

    async def start(self) -> None:
        """Start the Telegram bot with long polling."""
        if not self.config.token:
            logger.error("Telegram bot token not configured")
            return

        self._running = True

        # Build the application
        self._app = Application.builder().token(self.config.token).build()

        # Add message handler for text, photos, voice, documents
        self._app.add_handler(
            MessageHandler(
                (
                    filters.TEXT
                    | filters.PHOTO
                    | filters.VOICE
                    | filters.AUDIO
                    | filters.Document.ALL
                )
                & ~filters.COMMAND,
                self._on_message,
            )
        )

        # Add /start command handler
        from telegram.ext import CommandHandler

        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CallbackQueryHandler(self._on_callback_query))

        logger.info("Starting Telegram bot (polling mode)...")

        # Initialize and start polling
        await self._app.initialize()
        await self._app.start()

        # Get bot info
        bot_info = await self._app.bot.get_me()
        logger.info(f"Telegram bot @{bot_info.username} connected")

        # Start polling (this runs until stopped)
        await self._app.updater.start_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,  # Ignore old messages on startup
        )

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False

        if self._app:
            logger.info("Stopping Telegram bot...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Telegram."""
        if not self._app:
            logger.warning("Telegram bot not running")
            return

        # Delegate to edit() if this is an edit request
        if msg.edit_message_id:
            await self.edit(msg)
            return

        try:
            chat_id = int(msg.chat_id)

            # Clean up progress message for this chat
            progress_msg_id = self._progress_messages.pop(msg.chat_id, None)
            if progress_msg_id:
                try:
                    await self._app.bot.delete_message(chat_id=chat_id, message_id=progress_msg_id)
                except Exception:
                    pass

            # Convert markdown to Telegram HTML
            html_content = _markdown_to_telegram_html(msg.content)
            reply_markup = None
            if msg.components:
                reply_markup = self._build_telegram_keyboard(msg.components)
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=html_content,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
        except ValueError:
            logger.error(f"Invalid chat_id: {msg.chat_id}")
        except Exception as e:
            # Fallback to plain text if HTML parsing fails
            logger.warning(f"HTML parse failed, falling back to plain text: {e}")
            try:
                await self._app.bot.send_message(chat_id=int(msg.chat_id), text=msg.content)
            except Exception as e2:
                logger.error(f"Error sending Telegram message: {e2}")

    async def on_progress(self, event: ProgressEvent) -> None:
        """Display progress events via typing indicator and status messages."""
        if not self._app:
            return

        try:
            chat_id = int(event.chat_id)
            key = event.chat_id

            # Send typing action for all progress events
            await self._app.bot.send_chat_action(chat_id=chat_id, action="typing")

            # For multi-step processing, send/edit a status message
            if event.kind in (
                ProgressKind.TOOL_START,
                ProgressKind.TOOL_COMPLETE,
            ):
                if event.kind == ProgressKind.TOOL_START:
                    status = f"Running {event.tool_name}..."
                else:
                    status = f"{event.tool_name}: {event.detail}"

                if key in self._progress_messages:
                    try:
                        await self._app.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=self._progress_messages[key],
                            text=status,
                        )
                    except Exception:
                        self._progress_messages.pop(key, None)
                elif event.iteration > 1:
                    # Only show status message after first iteration
                    sent = await self._app.bot.send_message(chat_id=chat_id, text=status)
                    self._progress_messages[key] = sent.message_id

        except Exception as e:
            logger.debug(f"Telegram progress display failed: {e}")

    async def edit(self, msg: OutboundMessage) -> None:
        """Edit a previously sent Telegram message."""
        if not self._app or not msg.edit_message_id:
            return

        try:
            chat_id = int(msg.chat_id)
            html_content = _markdown_to_telegram_html(msg.content)
            await self._app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(msg.edit_message_id),
                text=html_content,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.warning(f"Failed to edit Telegram message: {e}")

    def _build_telegram_keyboard(self, components: list[dict]) -> InlineKeyboardMarkup:
        """Build Telegram inline keyboard from component definitions."""
        keyboard = []
        row: list[InlineKeyboardButton] = []

        for comp in components:
            if comp.get("type") == "button":
                row.append(
                    InlineKeyboardButton(
                        text=comp.get("label", ""),
                        callback_data=comp.get("callback_data", "")[:64],
                    )
                )
            elif comp.get("type") == "select":
                if row:
                    keyboard.append(row)
                    row = []
                for opt in comp.get("options", []):
                    keyboard.append(
                        [
                            InlineKeyboardButton(
                                text=opt.get("label", ""),
                                callback_data=opt.get("value", "")[:64],
                            )
                        ]
                    )

        if row:
            keyboard.append(row)

        return InlineKeyboardMarkup(keyboard)

    async def _on_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline keyboard callbacks."""
        query = update.callback_query
        if not query or not query.data:
            return

        await query.answer()

        user = update.effective_user
        chat_id = query.message.chat_id if query.message else 0
        sender_id = str(user.id) if user else "unknown"
        if user and user.username:
            sender_id = f"{sender_id}|{user.username}"

        await self.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                sender_id=sender_id,
                chat_id=str(chat_id),
                content=query.data,
            )
        )

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message or not update.effective_user:
            return

        user = update.effective_user
        await update.message.reply_text(
            f"👋 Hi {user.first_name}! I'm nanobot.\n\nSend me a message and I'll respond!"
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages (text, photos, voice, documents)."""
        if not update.message or not update.effective_user:
            return

        message = update.message
        user = update.effective_user
        chat_id = message.chat_id

        # Use stable numeric ID, but keep username for allowlist compatibility
        sender_id = str(user.id)
        if user.username:
            sender_id = f"{sender_id}|{user.username}"

        # Store chat_id for replies
        self._chat_ids[sender_id] = chat_id

        # Build content from text and/or media
        content_parts = []
        media_paths = []

        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)

        # Handle media files
        media_file = None
        media_type = None

        if message.photo:
            media_file = message.photo[-1]  # Largest photo
            media_type = "image"
        elif message.voice:
            media_file = message.voice
            media_type = "voice"
        elif message.audio:
            media_file = message.audio
            media_type = "audio"
        elif message.document:
            media_file = message.document
            media_type = "file"

        # Download media if present
        if media_file and self._app:
            try:
                file = await self._app.bot.get_file(media_file.file_id)
                ext = self._get_extension(media_type, getattr(media_file, "mime_type", None))

                # Save to workspace/media/
                from pathlib import Path

                media_dir = Path.home() / ".nanobot" / "media"
                media_dir.mkdir(parents=True, exist_ok=True)

                file_path = media_dir / f"{media_file.file_id[:16]}{ext}"
                await file.download_to_drive(str(file_path))

                media_paths.append(str(file_path))

                # Handle voice transcription
                if media_type == "voice" or media_type == "audio":
                    from nanobot.providers.transcription import GroqTranscriptionProvider

                    transcriber = GroqTranscriptionProvider(api_key=self.groq_api_key)
                    transcription = await transcriber.transcribe(file_path)
                    if transcription:
                        logger.info(f"Transcribed {media_type}: {transcription[:50]}...")
                        content_parts.append(f"[transcription: {transcription}]")
                    else:
                        content_parts.append(f"[{media_type}: {file_path}]")
                else:
                    content_parts.append(f"[{media_type}: {file_path}]")

                logger.debug(f"Downloaded {media_type} to {file_path}")
            except Exception as e:
                logger.error(f"Failed to download media: {e}")
                content_parts.append(f"[{media_type}: download failed]")

        content = "\n".join(content_parts) if content_parts else "[empty message]"

        logger.debug(f"Telegram message from {sender_id}: {content[:50]}...")

        # Forward to the message bus
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(chat_id),
            content=content,
            media=media_paths,
            metadata={
                "message_id": message.message_id,
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "is_group": message.chat.type != "private",
            },
        )

    def _get_extension(self, media_type: str, mime_type: str | None) -> str:
        """Get file extension based on media type."""
        if mime_type:
            ext_map = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "audio/ogg": ".ogg",
                "audio/mpeg": ".mp3",
                "audio/mp4": ".m4a",
            }
            if mime_type in ext_map:
                return ext_map[mime_type]

        type_map = {"image": ".jpg", "voice": ".ogg", "audio": ".mp3", "file": ""}
        return type_map.get(media_type, "")
