from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .config import ConfigError, HOME_CONFIG_PATH, load_telegram_config

_OCTOPUS = "\N{OCTOPUS}"


@dataclass(slots=True)
class SetupResult:
    missing_opencode: bool = False
    missing_or_invalid_config: bool = False
    config_path: Path = HOME_CONFIG_PATH

    @property
    def ok(self) -> bool:
        return not (self.missing_opencode or self.missing_or_invalid_config)


def check_setup() -> SetupResult:
    missing_opencode = shutil.which("opencode") is None

    try:
        config, config_path = load_telegram_config()
    except ConfigError:
        return SetupResult(
            missing_opencode=missing_opencode,
            missing_or_invalid_config=True,
            config_path=HOME_CONFIG_PATH,
        )

    token = config.get("bot_token")
    chat_id = config.get("chat_id")

    missing_or_invalid_config = not (isinstance(token, str) and token.strip())
    try:
        if chat_id is not None:
            int(chat_id)  # type: ignore
            chat_id_ok = True
        else:
            chat_id_ok = False
    except (ValueError, TypeError):
        chat_id_ok = False

    missing_or_invalid_config |= not chat_id_ok

    return SetupResult(
        missing_opencode=missing_opencode,
        missing_or_invalid_config=missing_or_invalid_config,
        config_path=config_path,
    )


def _config_path_display(path: Path) -> str:
    home = Path.home()
    try:
        return f"~/{path.relative_to(home)}"
    except ValueError:
        return str(path)


def render_setup_guide(result: SetupResult) -> None:
    if result.ok:
        return

    console = Console(stderr=True)
    parts: list[str] = []
    step = 0

    def add_step(title: str, *lines: str) -> None:
        nonlocal step
        step += 1
        parts.append(f"[bold yellow]{step}.[/] [bold]{title}[/]")
        parts.append("")
        parts.extend(lines)
        parts.append("")

    if result.missing_opencode:
        add_step(
            "Install the OpenCode CLI",
            "   [dim]$[/] npm install -g opencode",
        )

    if result.missing_or_invalid_config:
        config_display = _config_path_display(result.config_path)
        add_step(
            "Create a config",
            f"   [dim]{config_display}[/]",
            "",
            '   [cyan]bot_token[/] = [green]"123456789:ABCdef..."[/]',
            "   [cyan]chat_id[/]   = [green]123456789[/]",
            "",
            "[dim]" + ("-" * 56) + "[/]",
            "",
            "[bold]Getting your Telegram credentials:[/]",
            "",
            "   [cyan]bot_token[/]  create a bot with [link=https://t.me/BotFather]@BotFather[/]",
            "   [cyan]chat_id[/]    message [link=https://t.me/myidbot]@myidbot[/] to get your id",
        )

    panel = Panel(
        "\n".join(parts).rstrip(),
        title="[bold]Welcome to takopi![/]",
        subtitle=f"{_OCTOPUS} setup required",
        border_style="yellow",
        padding=(1, 2),
        expand=False,
    )
    console.print(panel)
