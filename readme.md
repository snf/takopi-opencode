# takopi

> **Note**: This is a fork of [takopi](https://github.com/banteg/takopi) by [banteg](https://github.com/banteg).
> While the original project targets Claude Code, this fork is adapted to work with [OpenCode](https://opencode.ai).

🐙 *A little helper from Happy Planet, here to make your OpenCode sessions happier-pi!*

A Telegram bot that bridges messages to [OpenCode](https://opencode.ai) sessions using non-interactive `opencode run --format json`.

## Features

- **Stateless Resume**: No database required—sessions are resumed via `resume: <uuid>` lines embedded in messages
- **Progress Updates**: Real-time progress edits showing commands, tools, and elapsed time
- **Markdown Rendering**: Full Telegram-compatible markdown with entity support
- **Concurrency**: Handles multiple conversations with per-session serialization
- **Token Redaction**: Automatically redacts Telegram tokens from logs

## Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager
- OpenCode CLI on PATH (`npm install -g opencode`)

### Installation

```bash
# Install with uv, then run as `takopi`
uv tool install takopi
takopi

# or run with uvx
uvx takopi
```

### Setup

1. **Start the bot**: Send `/start` to your bot in Telegram—it can't message you until you do
2. **Trust your working directory**: Run `codex` once interactively in your project directory (must be a git repo) to add it to trusted directories

### Configuration

Create `~/.config/takopi/config.toml` (or `.takopi/takopi.toml` for a repo-local config):

```toml
bot_token = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
chat_id = 123456789
```

| Key | Description |
|-----|-------------|
| `bot_token` | Telegram Bot API token from [@BotFather](https://t.me/BotFather) |
| `chat_id` | Your Telegram user ID from [@myidbot](https://t.me/myidbot) |

The bridge only accepts messages where the chat ID equals the sender ID and both match `chat_id` (i.e., private chat with that user).

### OpenCode Profile (Optional)

Create a profile in `~/.config/opencode/config.toml` if needed (details vary by OpenCode version).

Then run takopi with:

```bash
takopi --model claude-3-5-sonnet
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--final-notify` / `--no-final-notify` | `--final-notify` | Send final response as new message (vs. edit) |
| `--debug` / `--no-debug` | `--no-debug` | Enable verbose logging |
| `--model NAME` | (opencode default) | Model to use for the agent |
| `--version` |  | Show the version and exit |

## Usage

### New Conversation

Send any message to your bot. The bridge will:

1. Send a silent progress message
2. Stream events from `opencode run`
3. Update progress every ~2 seconds
4. Send final response with session ID

### Resume a Session

Reply to a bot message (containing `resume: <uuid>`), or include the resume line in your message:

```
resume: `019b66fc-64c2-7a71-81cd-081c504cfeb2`
```

### Cancel a Run

Reply to a progress message with `/cancel` to stop the running execution.

## Notes

- **Startup**: Pending updates are drained (ignored) on startup
- **Progress**: Updates are throttled to ~2s intervals, sent silently
- **Filtering**: Only accepts messages where chat ID equals sender ID and matches `chat_id`
- **Single instance**: Run exactly one instance per bot token—multiple instances will race for updates

## Development

See [`developing.md`](developing.md).

## License

MIT
