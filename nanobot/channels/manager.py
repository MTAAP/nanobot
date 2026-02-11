"""Channel manager for coordinating chat channels."""

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.delivery import DeliveryQueue
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Config


class ChannelManager:
    """
    Manages chat channels and coordinates message routing.

    Responsibilities:
    - Initialize enabled channels (Telegram, WhatsApp, etc.)
    - Start/stop channels
    - Route outbound messages with persistent delivery queue
    """

    def __init__(self, config: Config, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None

        # Persistent delivery queue for crash-safe outbound messages
        self._delivery_queue = DeliveryQueue(Path.home() / ".nanobot" / "data" / "delivery.db")
        self._delivery_queue.recover_in_flight()

        self._init_channels()
        self._wire_progress_subscriptions()

    def _init_channels(self) -> None:
        """Initialize channels based on config."""

        # Telegram channel
        if self.config.channels.telegram.enabled:
            try:
                from nanobot.channels.telegram import TelegramChannel

                self.channels["telegram"] = TelegramChannel(
                    self.config.channels.telegram,
                    self.bus,
                    groq_api_key=self.config.providers.groq.api_key,
                )
                logger.info("Telegram channel enabled")
            except ImportError as e:
                logger.warning(f"Telegram channel not available: {e}")

        # WhatsApp channel
        if self.config.channels.whatsapp.enabled:
            try:
                from nanobot.channels.whatsapp import WhatsAppChannel

                self.channels["whatsapp"] = WhatsAppChannel(self.config.channels.whatsapp, self.bus)
                logger.info("WhatsApp channel enabled")
            except ImportError as e:
                logger.warning(f"WhatsApp channel not available: {e}")

        # Feishu channel
        if self.config.channels.feishu.enabled:
            try:
                from nanobot.channels.feishu import FeishuChannel

                self.channels["feishu"] = FeishuChannel(self.config.channels.feishu, self.bus)
                logger.info("Feishu channel enabled")
            except ImportError as e:
                logger.warning(f"Feishu channel not available: {e}")

        # Discord channel
        if self.config.channels.discord.enabled:
            try:
                from nanobot.channels.discord import DiscordChannel

                self.channels["discord"] = DiscordChannel(
                    self.config.channels.discord,
                    self.bus,
                    groq_api_key=self.config.providers.groq.api_key,
                )
                logger.info("Discord channel enabled")
            except ImportError as e:
                logger.warning(f"Discord channel not available: {e}")

        # Web channel (dashboard chat)
        if self.config.web.enabled:
            try:
                from nanobot.channels.web import WebChannel

                self.channels["web"] = WebChannel(self.config.web, self.bus)
                logger.info("Web channel enabled")
            except ImportError as e:
                logger.warning(f"Web channel not available: {e}")

    def _wire_progress_subscriptions(self) -> None:
        """Subscribe each channel's on_progress to the message bus."""
        for name, channel in self.channels.items():
            self.bus.subscribe_progress(name, channel.on_progress)
            logger.debug(f"Wired progress subscription for {name}")

    async def start_all(self) -> None:
        """Start WhatsApp channel and the outbound dispatcher."""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        # Start outbound dispatcher
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        # Start WhatsApp channel
        tasks = []
        for name, channel in self.channels.items():
            logger.info(f"Starting {name} channel...")
            tasks.append(asyncio.create_task(channel.start()))

        # Wait for all to complete (they should run forever)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        logger.info("Stopping all channels...")

        # Stop dispatcher
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        # Stop all channels
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info(f"Stopped {name} channel")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")

        # Close delivery queue
        self._delivery_queue.close()

    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages with persistent delivery and retry."""
        logger.info("Outbound dispatcher started (with delivery queue)")

        while True:
            try:
                # 1. Check for new messages from bus
                try:
                    msg = await asyncio.wait_for(self.bus.consume_outbound(), timeout=0.5)
                    self._delivery_queue.enqueue(msg)
                except asyncio.TimeoutError:
                    pass

                # 2. Process ready messages from delivery queue
                ready = self._delivery_queue.dequeue_ready()
                for delivery_id, msg in ready:
                    channel = self.channels.get(msg.channel)
                    if not channel:
                        self._delivery_queue.mark_failed(
                            delivery_id,
                            f"unknown channel: {msg.channel}",
                        )
                        continue
                    try:
                        await channel.send(msg)
                        self._delivery_queue.mark_delivered(delivery_id)
                    except Exception as e:
                        self._delivery_queue.mark_failed(delivery_id, str(e)[:500])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dispatch error: {e}")
                await asyncio.sleep(1)

    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all channels."""
        return {
            name: {"enabled": True, "running": channel.is_running}
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        """Get list of enabled channel names."""
        return list(self.channels.keys())
