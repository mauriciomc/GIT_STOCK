# GIT_STOCK
Stock analysis algorithm written in Python

Install the library requirements for running the trader and analysis tool:
pip install -r requirements.txt

## Configuration and secrets

`config/config.json` holds the shareable settings (strategy, parameters, sectors/tickers) and is
committed with the Telegram fields left empty. Put your own bot credentials in a private file:

```bash
cp config/secrets.example.json config/secrets.json   # git-ignored
chmod 600 config/secrets.json
# edit: token from @BotFather, your chat_id, optional bot_name
```

`Config.load()` overlays `config/secrets.json` (or the file in `$GIT_STOCK_SECRETS`) on top of the
config, and the env vars `GIT_STOCK_TELEGRAM_TOKEN` / `GIT_STOCK_TELEGRAM_CHAT_ID` override both.
Market data (`stock_data/`, `stock_data_full/`, `data_cache/`) and databases are git-ignored;
they are downloaded/rebuilt locally.
