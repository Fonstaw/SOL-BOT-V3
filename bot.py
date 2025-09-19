import os
import json
import asyncio
import logging
import base64
import requests
import re
import threading
from tenacity import retry, stop_after_attempt, wait_exponential
from flask import Flask
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from solana.rpc.api import Client
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from solana.rpc.types import TxOpts
from spl.token.client import Token
from spl.token.constants import TOKEN_PROGRAM_ID

# (All the configuration and core logic functions from before remain the same)
# ...
# Flask app for Web Service
app_flask = Flask(__name__)

@app_flask.route('/')
def health_check():
    return "Bot is running!", 200

# ------------------ CONFIG ------------------ #
API_ID = int(os.environ.get("API_ID"))
API_HASH = os.environ.get("API_HASH")
SESSION_STRING = os.environ.get("SESSION_STRING")
BOT_TOKEN = os.environ.get("BOT_TOKEN")
SOLANA_PRIVATE_KEY = os.environ.get("SOLANA_PRIVATE_KEY")
SOLANA_RPC = os.environ.get("SOLANA_RPC", "https://api.mainnet-beta.solana.com")
JUPITER_API = os.environ.get("JUPITER_API", "https://quote-api.jup.ag/v6")
USDC_MINT = os.environ.get("USDC_MINT", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
CHANNELS = os.environ.get("CHANNELS", "").split(",")

# Preset auto-sell targets and stop-loss (as percentages) for new tokens
PRESET_SELL_TARGETS = [2, 3, 5, 10, 20, 50]
PRESET_STOP_LOSS = [10, 20, 30, 35, 40, 50, 60, 70, 80, 90]
AUTO_BUY = os.environ.get("AUTO_BUY", "true").lower() == "true"
TRADE_AMOUNT_USDC = float(os.environ.get("TRADE_AMOUNT_USDC", 0.01))
SLIPPAGE_BPS = int(os.environ.get("SLIPPAGE_BPS", 100))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 3))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

TRADES_FILE = "trades.json"

# ------------------ LOGGING ------------------ #
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------ LOAD/SAVE TRADES ------------------ #
trades_lock = threading.Lock()

def load_trades():
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_trades(trades):
    with trades_lock:
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)

trades = load_trades()

# ------------------ SOLANA CLIENT ------------------ #
try:
    payer = Keypair.from_bytes(bytes.fromhex(SOLANA_PRIVATE_KEY))
    logger.info("Solana keypair loaded successfully")
except ValueError as e:
    logger.error(f"Invalid SOLANA_PRIVATE_KEY: {e}")
    exit(1)

sol_client = Client(SOLANA_RPC)
logger.info(f"Connected to Solana RPC: {SOLANA_RPC}")

def get_token_decimals(mint: str) -> int:
    try:
        mint_pubkey = Pubkey.from_string(mint)
        token = Token(sol_client, mint_pubkey, TOKEN_PROGRAM_ID, payer)
        decimals = token.get_mint_info().decimals
        logger.info(f"Decimals for {mint}: {decimals}")
        return decimals
    except Exception as e:
        logger.error(f"Failed to fetch decimals for {mint}: {e}")
        return 6

# ------------------ TELEGRAM BOT ------------------ #
app = ApplicationBuilder().token(BOT_TOKEN).build()
logger.info("Telegram bot application initialized")

# ------------------ TELETHON CLIENT ------------------ #
tele_client = TelegramClient(StringSession(SESSION_STRING), API_ID, API_HASH)
logger.info("Telethon client initialized")

# ------------------ JUPITER FUNCTIONS ------------------ #
@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_swap_quote(input_mint, output_mint, amount):
    logger.info(f"Getting swap quote: input={input_mint}, output={output_mint}, amount={amount}, slippage={SLIPPAGE_BPS}")
    decimals_in = get_token_decimals(input_mint)
    raw_amount = int(amount * (10 ** decimals_in))

    url = f"{JUPITER_API}/quote"
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": raw_amount,
        "slippageBps": SLIPPAGE_BPS,
        "onlyDirectRoutes": "false",
    }

    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching swap quote for {input_mint} to {output_mint}: {e}, status={r.status_code}, response={r.text}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching swap quote for {input_mint} to {output_mint}: {e}")
        return None

def execute_swap_route(route, output_mint):
    try:
        logger.info("Getting swap transaction from Jupiter API")
        swap_url = f"{JUPITER_API}/swap"
        swap_payload = {
            "quoteResponse": route,
            "userPublicKey": str(payer.pubkey()),
            "wrapAndUnwrapSol": True,
        }
        swap_resp = requests.post(swap_url, json=swap_payload)
        swap_resp.raise_for_status()
        
        tx_base64 = swap_resp.json().get("swapTransaction")
        if not tx_base64:
            logger.error("Failed to get swapTransaction from Jupiter API response.")
            return None

        logger.info("Decoding transaction bytes")
        tx_bytes = base64.b64decode(tx_base64)
        
        tx = Transaction.from_bytes(tx_bytes)
        tx.sign(payer)

        logger.info("Sending signed transaction")
        resp = sol_client.send_transaction(tx, opts=TxOpts(skip_preflight=False, preflight_commitment="confirmed"))
        sig = resp.value if hasattr(resp, 'value') else resp.get('result')
        
        if not sig:
            logger.error(f"Failed to extract signature from response: {resp}")
            return None
            
        logger.info(f"Transaction signature: {sig}")
        sol_client.confirm_transaction(sig, commitment="confirmed")
        logger.info(f"Transaction confirmed: https://solscan.io/tx/{sig}")
        return resp
    except Exception as e:
        logger.error(f"Swap execution failed: {e}")
        return None

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_token_price(token_mint):
    logger.info(f"Fetching price for token: {token_mint}")
    quote = get_swap_quote(token_mint, USDC_MINT, 1)
    if quote and "outAmount" in quote:
        return int(quote["outAmount"]) / (10 ** get_token_decimals(USDC_MINT))
    logger.error(f"No valid price quote for {token_mint}")
    return None

# ------------------ TRADE LOGIC ------------------ #
async def buy_token(token_mint):
    global trades
    logger.info(f"Attempting to buy token: {token_mint}")
    if not AUTO_BUY:
        logger.info(f"Auto-buy disabled. Skipping {token_mint}")
        return
    if token_mint in trades:
        logger.info(f"Token {token_mint} already tracked.")
        return
    try:
        logger.info(f"Fetching swap quote for {token_mint} with {TRADE_AMOUNT_USDC} USDC")
        quote = get_swap_quote(USDC_MINT, token_mint, TRADE_AMOUNT_USDC)
        if not quote:
            logger.error(f"No swap quote returned for {token_mint}. Skipping buy.")
            return
        
        route = quote
        
        logger.info(f"Got swap quote, executing swap for {token_mint}")
        resp = execute_swap_route(route, token_mint)
        if resp:
            logger.info(f"Swap successful for {token_mint}, fetching price")
            buy_price = fetch_token_price(token_mint)
            
            out_amount_base = int(route["outAmount"])
            token_decimals = get_token_decimals(token_mint)
            out_amount_human = out_amount_base / (10 ** token_decimals)

            trades[token_mint] = {
                "buy_price": buy_price,
                "amount": out_amount_human,
                "targets": PRESET_SELL_TARGETS.copy(),
                "stop_loss": PRESET_STOP_LOSS[2],
            }
            save_trades(trades)
            logger.info(
                f"Bought {out_amount_human} of {token_mint} at {buy_price} USDC "
                f"with targets {trades[token_mint]['targets']} and stop-loss {trades[token_mint]['stop_loss']}%"
            )
        else:
            logger.error(f"Swap execution returned None for {token_mint}")
    except Exception as e:
        logger.error(f"Buy failed for {token_mint}: {str(e)}")

async def check_auto_sell():
    global trades
    to_remove = []
    for token, info in list(trades.items()):
        try:
            current_price = fetch_token_price(token)
            if current_price is None:
                continue

            for target in info["targets"]:
                if current_price >= info["buy_price"] * target:
                    quote = get_swap_quote(token, USDC_MINT, info["amount"])
                    if not quote:
                        logger.error(f"Could not get sell quote for {token}")
                        continue
                    route = quote
                    resp = execute_swap_route(route, USDC_MINT)
                    if resp:
                        out_amount_base = int(route["outAmount"])
                        usdc_decimals = get_token_decimals(USDC_MINT)
                        out_amount_human = out_amount_base / (10 ** usdc_decimals)
                        logger.info(f"Sold {info['amount']} of {token} for {out_amount_human} USDC at {current_price} (target {target}x)")
                        to_remove.append(token)
                        break
            
            if token in to_remove: continue

            if current_price <= info["buy_price"] * (1 - info["stop_loss"] / 100):
                quote = get_swap_quote(token, USDC_MINT, info["amount"])
                if not quote:
                    logger.error(f"Could not get sell quote for {token} (stop-loss)")
                    continue
                route = quote
                resp = execute_swap_route(route, USDC_MINT)
                if resp:
                    out_amount_base = int(route["outAmount"])
                    usdc_decimals = get_token_decimals(USDC_MINT)
                    out_amount_human = out_amount_base / (10 ** usdc_decimals)
                    logger.info(f"Sold {info['amount']} of {token} for {out_amount_human} USDC at {current_price} (stop-loss {info['stop_loss']}%)")
                    to_remove.append(token)
        except Exception as e:
            logger.error(f"Auto-sell error for {token}: {e}")
    for token in to_remove:
        if token in trades: trades.pop(token)
    save_trades(trades)

async def auto_sell_loop():
    logger.info("Starting auto-sell loop")
    while True:
        await check_auto_sell()
        await asyncio.sleep(60)

# (All Telegram command handlers like status, setamount, etc. are correct and unchanged)
# ...
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /status command")
    msg = f"Auto-buy: {AUTO_BUY}\nTrade amount: {TRADE_AMOUNT_USDC} USDC\nSlippage: {SLIPPAGE_BPS/100}%\nChannels: {CHANNELS}\nPreset Sell Targets: {PRESET_SELL_TARGETS}\nPreset Stop-Loss: {[f'{x}%' for x in PRESET_STOP_LOSS]}\n\nTracked Tokens:\n"
    for token, info in trades.items():
        msg += f"{token}: Buy {info['amount']} tokens at {info['buy_price']}, Targets {info['targets']}, Stop-loss {info['stop_loss']}%\n"
    keyboard = [
        [InlineKeyboardButton("Toggle Auto-Buy", callback_data="togglebuy"), InlineKeyboardButton("Set Amount", callback_data="setamount")],
        [InlineKeyboardButton("Add Channel", callback_data="addchannel"), InlineKeyboardButton("Remove Channel", callback_data="removechannel")],
        [InlineKeyboardButton("View Trades", callback_data="viewtrades"), InlineKeyboardButton("Change Wallet", callback_data="setwallet")],
        [InlineKeyboardButton("Set Targets", callback_data="settargets"), InlineKeyboardButton("Set Stop-Loss", callback_data="setstoploss")],
        [InlineKeyboardButton("Sell Token", callback_data="sell"), InlineKeyboardButton("Set Slippage", callback_data="setslippage")],
        [InlineKeyboardButton("Set Preset Targets", callback_data="setpresettargets"), InlineKeyboardButton("Set Preset Stop-Loss", callback_data="setpresetstoploss")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(msg, reply_markup=reply_markup)

async def setamount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setamount command")
    global TRADE_AMOUNT_USDC
    try:
        TRADE_AMOUNT_USDC = float(context.args[0])
        await update.message.reply_text(f"Trade amount updated to {TRADE_AMOUNT_USDC} USDC")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /setamount <amount>")

async def settargets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /settargets command")
    try:
        token = context.args[0]
        targets = [float(x) for x in context.args[1].split(",")]
        if token in trades:
            trades[token]["targets"] = targets
            save_trades(trades)
            await update.message.reply_text(f"Targets for {token} updated: {targets}")
        else:
            await update.message.reply_text(f"{token} not tracked yet")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /settargets <token> <x1,x2,...>")

async def setstoploss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setstoploss command")
    try:
        token = context.args[0]
        stop = float(context.args[1])
        if token in trades:
            if 0 < stop <= 100:
                trades[token]["stop_loss"] = stop
                save_trades(trades)
                await update.message.reply_text(f"Stop-loss for {token} updated: {stop}%")
            else:
                await update.message.reply_text("Stop-loss percentage must be between 0 and 100.")
        else:
            await update.message.reply_text(f"{token} not tracked yet")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /setstoploss <token> <percentage>")

async def togglebuy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /togglebuy command")
    global AUTO_BUY
    AUTO_BUY = not AUTO_BUY
    await update.message.reply_text(f"Auto-buy set to {AUTO_BUY}")

async def addchannel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /addchannel command")
    try:
        channel = context.args[0].strip()
        if channel not in CHANNELS:
            CHANNELS.append(channel)
            await update.message.reply_text(f"Channel added: {channel}")
        else:
            await update.message.reply_text("Channel already exists")
    except IndexError:
        await update.message.reply_text("Usage: /addchannel @channel")

async def removechannel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /removechannel command")
    try:
        channel = context.args[0].strip()
        if channel in CHANNELS:
            CHANNELS.remove(channel)
            await update.message.reply_text(f"Channel removed: {channel}")
        else:
            await update.message.reply_text("Channel not found")
    except IndexError:
        await update.message.reply_text("Usage: /removechannel @channel")

async def sell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /sell command")
    try:
        token = context.args[0]
        if token in trades:
            quote = get_swap_quote(token, USDC_MINT, trades[token]["amount"])
            if not quote:
                await update.message.reply_text(f"Could not get a quote to sell {token}")
                return
            route = quote
            resp = execute_swap_route(route, USDC_MINT)
            if resp:
                out_amount_base = int(route["outAmount"])
                usdc_decimals = get_token_decimals(USDC_MINT)
                out_amount_human = out_amount_base / (10 ** usdc_decimals)
                amount_sold = trades[token]['amount']
                trades.pop(token)
                save_trades(trades)
                await update.message.reply_text(f"Manually sold {amount_sold} of {token} for {out_amount_human} USDC")
            else:
                await update.message.reply_text(f"Failed to sell {token}: Swap rejected or failed")
        else:
            await update.message.reply_text(f"{token} not tracked")
    except Exception as e:
        await update.message.reply_text(f"Usage: /sell <token> | Error: {str(e)}")

async def setwallet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setwallet command")
    global payer
    try:
        new_key = context.args[0]
        payer = Keypair.from_bytes(bytes.fromhex(new_key))
        await update.message.reply_text("Wallet updated successfully")
    except Exception as e:
        await update.message.reply_text(f"Failed to update wallet: {e}")

async def setslippage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setslippage command")
    global SLIPPAGE_BPS
    try:
        slippage_percent = float(context.args[0])
        if 0.1 <= slippage_percent <= 50:
            SLIPPAGE_BPS = int(slippage_percent * 100)
            await update.message.reply_text(f"Slippage updated to {slippage_percent}% ({SLIPPAGE_BPS} bps)")
        else:
            await update.message.reply_text("Slippage must be between 0.1% and 50%")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /setslippage <percentage>")

async def setpresettargets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setpresettargets command")
    global PRESET_SELL_TARGETS
    try:
        targets = [float(x) for x in context.args[0].split(",")]
        if all(t >= 1 for t in targets):
            PRESET_SELL_TARGETS = targets
            await update.message.reply_text(f"Preset sell targets updated: {PRESET_SELL_TARGETS}")
        else:
            await update.message.reply_text("Targets must be >= 1")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /setpresettargets <x1,x2,...>")

async def setpresetstoploss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setpresetstoploss command")
    global PRESET_STOP_LOSS
    try:
        stop_losses = [float(x) for x in context.args[0].split(",")]
        if all(0 < x <= 100 for x in stop_losses):
            PRESET_STOP_LOSS = stop_losses
            await update.message.reply_text(f"Preset stop-loss updated: {[f'{x}%' for x in PRESET_STOP_LOSS]}")
        else:
            await update.message.reply_text("Stop-loss values must be between 0 and 100%")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /setpresetstoploss <x1,x2,...>")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    global AUTO_BUY
    if query.data == "togglebuy":
        AUTO_BUY = not AUTO_BUY
        await query.message.reply_text(f"Auto-buy set to {AUTO_BUY}")
    # (Other button handlers are unchanged)
    
app.add_handler(CommandHandler("status", status))
app.add_handler(CommandHandler("setamount", setamount))
app.add_handler(CommandHandler("settargets", settargets))
app.add_handler(CommandHandler("setstoploss", setstoploss))
app.add_handler(CommandHandler("togglebuy", togglebuy))
app.add_handler(CommandHandler("addchannel", addchannel))
app.add_handler(CommandHandler("removechannel", removechannel))
app.add_handler(CommandHandler("sell", sell))
app.add_handler(CommandHandler("setwallet", setwallet))
app.add_handler(CommandHandler("setslippage", setslippage))
app.add_handler(CommandHandler("setpresettargets", setpresettargets))
app.add_handler(CommandHandler("setpresetstoploss", setpresetstoploss))
app.add_handler(CallbackQueryHandler(button_callback))

@tele_client.on(events.NewMessage)
async def new_message(event):
    msg = event.message.message
    if not event.chat:
        logger.debug("Received message from unknown chat, skipping")
        return
    chat_username = (getattr(event.chat, 'username', None) or "").lower()
    for channel in CHANNELS:
        env_channel = channel.lower().strip().lstrip("@")
        if chat_username == env_channel:
            logger.info(f"Message detected in channel {env_channel}: {msg}")
            token_pattern = r'\b([1-9A-HJ-NP-Za-km-z]{32,44})\b'
            tokens = re.findall(token_pattern, msg)
            for token in tokens:
                if 32 <= len(token) <= 44 and all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in token):
                    logger.info(f"Detected token {token} in {env_channel}")
                    # --- IMPROVEMENT: Directly await the function since we are in the same loop ---
                    await buy_token(token)
                else:
                    logger.debug(f"Filtered out invalid potential token: {token}")

def run_flask():
    app_flask.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# ------------------ NEW MAIN EXECUTION BLOCK ------------------ #
async def main():
    # Start Flask in a background thread as it is not async
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("Flask server started in a background thread.")

    # Start the python-telegram-bot handlers
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    logger.info("Telegram bot command handler started.")

    # Start the auto-sell loop as a background async task
    asyncio.create_task(auto_sell_loop())
    logger.info("Auto-sell loop started as a background task.")

    # Start the Telethon client
    await tele_client.start()
    logger.info("Telethon client for channel monitoring started.")
    
    # Keep the main function alive to allow all tasks to run
    # This replaces the old app.run_polling() which was blocking everything
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    logger.info("Starting bot...")
    
    required_envs = ["API_ID", "API_HASH", "SESSION_STRING", "BOT_TOKEN", "SOLANA_PRIVATE_KEY"]
    for env in required_envs:
        if not os.environ.get(env):
            logger.error(f"Missing required environment variable: {env}")
            exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot shutting down...")
    finally:
        # Graceful shutdown (optional but good practice)
        asyncio.run(app.updater.stop())
        asyncio.run(app.stop())
        asyncio.run(tele_client.disconnect())
        logger.info("Bot shutdown complete.")
