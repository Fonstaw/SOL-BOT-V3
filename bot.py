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
from solana.rpc.types import Transaction
from solana.rpc.types import TxOpts
from spl.token.client import Token
from spl.token.constants import TOKEN_PROGRAM_ID

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
JUPITER_API = os.environ.get("JUPITER_API", "https://quote-api.jup.ag/v4")
USDC_MINT = os.environ.get("USDC_MINT", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
CHANNELS = os.environ.get("CHANNELS", "").split(",")

# Preset auto-sell targets and stop-loss (as percentages) for new tokens
PRESET_SELL_TARGETS = [2, 3, 5, 10, 20, 50]
PRESET_STOP_LOSS = [10, 20, 30, 35, 40, 50, 60, 70, 80, 90]  # Percentages (10% = 0.9 multiplier, etc.)
AUTO_SELL_TARGETS = [float(x) for x in os.environ.get("AUTO_SELL_TARGETS", "2,3,5").split(",")]
AUTO_BUY = os.environ.get("AUTO_BUY", "true").lower() == "true"
TRADE_AMOUNT_USDC = float(os.environ.get("TRADE_AMOUNT_USDC", 0.01))
SLIPPAGE_BPS = int(os.environ.get("SLIPPAGE_BPS", 100))  # Default 1% (100 basis points)
MIN_SLIPPAGE_TOLERANCE = float(os.environ.get("MIN_SLIPPAGE_TOLERANCE", 1.0))  # Default 1% for min-receive
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
    """
    Return decimals for an SPL token mint address using spl.token client.
    Falls back to 6 if it cannot fetch.
    """
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
    """
    Ask Jupiter for a swap quote. `amount` is in human units (e.g. 2.0 USDC).
    We convert to the smallest unit using the token's decimals.
    """
    logger.info(f"Getting swap quote: input={input_mint}, output={output_mint}, amount={amount}, slippage={SLIPPAGE_BPS}")

    # Convert the human amount to the smallest unit for the input token
    decimals_in = get_token_decimals(input_mint)
    raw_amount = int(amount * (10 ** decimals_in))

    url = f"{JUPITER_API}/quote"
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": raw_amount,
        "slippageBps": SLIPPAGE_BPS,
        "onlyDirectRoutes": False,
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def execute_swap_route(route, output_mint, expected_out_amount):
    try:
        # Check minimum receive amount
        out_amount_base = int(route["outAmount"])
        min_amount_base = int(expected_out_amount * (1 - MIN_SLIPPAGE_TOLERANCE / 100))
        if out_amount_base < min_amount_base:
            logger.error(f"Swap rejected: outAmount {out_amount_base} below min {min_amount_base} for {output_mint}")
            return None

        logger.info("Decoding transaction bytes")
        tx_bytes = base64.b64decode(route["swapTransaction"])

        # 1. Deserialize into a Transaction object
        tx = Transaction.deserialize(tx_bytes)

        # 2. Sign with your payer keypair
        tx.sign(payer)

        # 3. Send transaction and wait for confirmation
        logger.info("Sending signed transaction")
        resp = sol_client.send_transaction(tx, payer, opts=TxOpts(skip_preflight=False, preflight_commitment="confirmed"))

        sig = resp["result"] if "result" in resp else resp
        logger.info(f"Transaction signature: {sig}")

        # 4. Confirm transaction
        sol_client.confirm_transaction(sig, commitment="confirmed")
        logger.info(f"Transaction confirmed: {sig}")
        return resp
    except Exception as e:
        logger.error(f"Swap execution failed: {str(e)}")
        return None

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_token_price(token_mint):
    logger.info(f"Fetching price for token: {token_mint}")
    quote = get_swap_quote(token_mint, USDC_MINT, 1)
    if quote and quote.get("data"):
        return quote["data"][0]["outAmount"] / (10 ** get_token_decimals(USDC_MINT))
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
        if quote and quote.get("data"):
            route = quote["data"][0]
            logger.info(f"Got swap quote: {route}")
            logger.info(f"Executing swap for {token_mint}")
            expected_out_amount = int(route["outAmount"])
            resp = execute_swap_route(route, token_mint, expected_out_amount)
            if resp:
                logger.info(f"Swap successful for {token_mint}, fetching price")
                buy_price = fetch_token_price(token_mint)

                # NEW: how many tokens did we actually receive?
                out_amount_base = int(route["outAmount"])  # base units returned by Jupiter
                token_decimals = get_token_decimals(token_mint)  # your decimals helper
                out_amount_human = out_amount_base / (10 ** token_decimals)

                trades[token_mint] = {
                    "buy_price": buy_price,          # price in USDC
                    "amount": out_amount_human,      # store token quantity you now hold
                    "targets": PRESET_SELL_TARGETS.copy(),
                    "stop_loss": PRESET_STOP_LOSS[2],  # Default to 30%
                }
                save_trades(trades)
                logger.info(
                    f"Bought {out_amount_human} of {token_mint} at {buy_price} USDC "
                    f"with targets {trades[token_mint]['targets']} and stop-loss {trades[token_mint]['stop_loss']}%"
                )
            else:
                logger.error(f"Swap execution returned None for {token_mint}")
        else:
            logger.error(f"No valid swap quote for {token_mint}: {quote}")
    except Exception as e:
        logger.error(f"Buy failed for {token_mint}: {str(e)}")

async def check_auto_sell():
    global trades
    to_remove = []
    for token, info in trades.items():
        try:
            current_price = fetch_token_price(token)
            if current_price is None:
                continue
            for target in info["targets"]:
                if current_price >= info["buy_price"] * target:
                    quote = get_swap_quote(token, USDC_MINT, info["amount"])
                    route = quote["data"][0]
                    expected_out_amount = int(route["outAmount"])
                    resp = execute_swap_route(route, USDC_MINT, expected_out_amount)
                    if resp:
                        out_amount_base = int(route["outAmount"])
                        usdc_decimals = get_token_decimals(USDC_MINT)
                        out_amount_human = out_amount_base / (10 ** usdc_decimals)
                        logger.info(f"Sold {info['amount']} of {token} for {out_amount_human} USDC at {current_price} (target {target}x)")
                        to_remove.append(token)
                    break
                if current_price <= info["buy_price"] * (1 - info["stop_loss"] / 100):
                    quote = get_swap_quote(token, USDC_MINT, info["amount"])
                    route = quote["data"][0]
                    expected_out_amount = int(route["outAmount"])
                    resp = execute_swap_route(route, USDC_MINT, expected_out_amount)
                    if resp:
                        out_amount_base = int(route["outAmount"])
                        usdc_decimals = get_token_decimals(USDC_MINT)
                        out_amount_human = out_amount_base / (10 ** usdc_decimals)
                        logger.info(f"Sold {info['amount']} of {token} for {out_amount_human} USDC at {current_price} (stop-loss {info['stop_loss']}%)")
                        to_remove.append(token)
                    break
        except Exception as e:
            logger.error(f"Auto-sell error for {token}: {e}")
    for token in to_remove:
        trades.pop(token)
    save_trades(trades)

async def auto_sell_loop():
    logger.info("Starting auto-sell loop")
    while True:
        await check_auto_sell()
        await asyncio.sleep(60)

# ------------------ TELEGRAM COMMANDS ------------------ #
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /status command")
    msg = f"Auto-buy: {AUTO_BUY}\nTrade amount: {TRADE_AMOUNT_USDC} USDC\nSlippage: {SLIPPAGE_BPS/100}%\nMin Slippage Tolerance: {MIN_SLIPPAGE_TOLERANCE}%\nChannels: {CHANNELS}\nPreset Sell Targets: {PRESET_SELL_TARGETS}\nPreset Stop-Loss: {[f'{x}%' for x in PRESET_STOP_LOSS]}\n\nTracked Tokens:\n"
    for token, info in trades.items():
        msg += f"{token}: Buy {info['amount']} tokens at {info['buy_price']}, Targets {info['targets']}, Stop-loss {info['stop_loss']}%\n"
    keyboard = [
        [
            InlineKeyboardButton("Toggle Auto-Buy", callback_data="togglebuy"),
            InlineKeyboardButton("Set Amount", callback_data="setamount"),
        ],
        [
            InlineKeyboardButton("Add Channel", callback_data="addchannel"),
            InlineKeyboardButton("Remove Channel", callback_data="removechannel"),
        ],
        [
            InlineKeyboardButton("View Trades", callback_data="viewtrades"),
            InlineKeyboardButton("Change Wallet", callback_data="setwallet"),
        ],
        [
            InlineKeyboardButton("Set Targets", callback_data="settargets"),
            InlineKeyboardButton("Set Stop-Loss", callback_data="setstoploss"),
        ],
        [
            InlineKeyboardButton("Sell Token", callback_data="sell"),
            InlineKeyboardButton("Set Slippage", callback_data="setslippage"),
        ],
        [
            InlineKeyboardButton("Set Preset Targets", callback_data="setpresettargets"),
            InlineKeyboardButton("Set Preset Stop-Loss", callback_data="setpresetstoploss"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(msg, reply_markup=reply_markup)

async def setamount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setamount command")
    global TRADE_AMOUNT_USDC
    try:
        TRADE_AMOUNT_USDC = float(context.args[0])
        await update.message.reply_text(f"Trade amount updated to {TRADE_AMOUNT_USDC} USDC")
    except:
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
    except:
        await update.message.reply_text("Usage: /settargets <token> <x1,x2,...>")

async def setstoploss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setstoploss command")
    try:
        token = context.args[0]
        stop = float(context.args[1])
        if 0 < stop <= 100:
            trades[token]["stop_loss"] = stop
            save_trades(trades)
            await update.message.reply_text(f"Stop-loss for {token} updated: {stop}%")
        else:
            await update.message.reply_text(f"{token} not tracked yet")
    except:
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
    except:
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
    except:
        await update.message.reply_text("Usage: /removechannel @channel")

async def sell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /sell command")
    try:
        token = context.args[0]
        if token in trades:
            quote = get_swap_quote(token, USDC_MINT, trades[token]["amount"])
            route = quote["data"][0]
            expected_out_amount = int(route["outAmount"])
            resp = execute_swap_route(route, USDC_MINT, expected_out_amount)
            if resp:
                out_amount_base = int(route["outAmount"])
                usdc_decimals = get_token_decimals(USDC_MINT)
                out_amount_human = out_amount_base / (10 ** usdc_decimals)
                trades.pop(token)
                save_trades(trades)
                await update.message.reply_text(f"Manually sold {trades[token]['amount']} of {token} for {out_amount_human} USDC")
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
        key_bytes = bytes.fromhex(new_key)
        payer = Keypair.from_bytes(key_bytes)
        await update.message.reply_text("Wallet updated successfully")
    except Exception as e:
        await update.message.reply_text(f"Failed to update wallet: {e}")

async def setslippage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setslippage command")
    global SLIPPAGE_BPS, MIN_SLIPPAGE_TOLERANCE
    try:
        slippage_percent = float(context.args[0])
        if 0.1 <= slippage_percent <= 10:
            SLIPPAGE_BPS = int(slippage_percent * 100)
            MIN_SLIPPAGE_TOLERANCE = slippage_percent  # Sync with SLIPPAGE_BPS
            await update.message.reply_text(f"Slippage and min slippage tolerance updated to {slippage_percent}% ({SLIPPAGE_BPS} bps)")
        else:
            await update.message.reply_text("Slippage must be between 0.1% and 10%")
    except:
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
    except:
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
    except:
        await update.message.reply_text("Usage: /setpresetstoploss <x1,x2,...>")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    global AUTO_BUY, SLIPPAGE_BPS, PRESET_SELL_TARGETS, PRESET_STOP_LOSS, MIN_SLIPPAGE_TOLERANCE
    if query.data == "togglebuy":
        AUTO_BUY = not AUTO_BUY
        await query.message.reply_text(f"Auto-buy set to {AUTO_BUY}")
    elif query.data == "setamount":
        await query.message.reply_text("Send /setamount <amount> to set trade amount")
    elif query.data == "addchannel":
        await query.message.reply_text("Send /addchannel @channelname to add a channel")
    elif query.data == "removechannel":
        await query.message.reply_text("Send /removechannel @channelname to remove a channel")
    elif query.data == "viewtrades":
        msg = "Tracked Tokens:\n"
        for token, info in trades.items():
            msg += f"{token}: Buy {info['amount']} tokens at {info['buy_price']}\n"
        await query.message.reply_text(msg or "No trades yet")
    elif query.data == "setwallet":
        await query.message.reply_text("Send /setwallet <128-char-hex-key> to change wallet")
    elif query.data == "settargets":
        await query.message.reply_text("Send /settargets <token> <x1,x2,...> to set sell targets")
    elif query.data == "setstoploss":
        await query.message.reply_text("Send /setstoploss <token> <percentage> to set stop-loss")
    elif query.data == "sell":
        await query.message.reply_text("Send /sell <token> to sell a tracked token")
    elif query.data == "setslippage":
        await query.message.reply_text("Send /setslippage <percentage> to set slippage (0.1 to 10)")
    elif query.data == "setpresettargets":
        await query.message.reply_text("Send /setpresettargets <x1,x2,...> to set preset sell targets")
    elif query.data == "setpresetstoploss":
        await query.message.reply_text("Send /setpresetstoploss <x1,x2,...> to set preset stop-loss")

# Register commands
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

# ------------------ TELETHON LISTENER ------------------ #
@tele_client.on(events.NewMessage)
async def new_message(event):
    msg = event.message.message
    # Safety check - ensure chat exists
    if not event.chat:
        logger.debug("Received message from unknown chat, skipping")
        return
    chat_username = (event.chat.username or "").lower()
    for channel in CHANNELS:
        env_channel = channel.lower().strip().lstrip("@")
        if chat_username == env_channel:
            logger.info(f"Message detected in channel {env_channel}: {msg}")
            token_pattern = r"[1-9A-HJ-NP-Za-km-z]{44}"
            tokens = re.findall(token_pattern, msg)
            for token in tokens:
                try:
                    base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
                    if len(token) == 44 and all(c in base58_chars for c in token):
                        logger.info(f"Detected token {token} in {env_channel}")
                        await buy_token(token)
                except Exception as e:
                    logger.error(f"Error with token {token}: {e}")

# ------------------ MAIN EXECUTION ------------------ #
def run_flask():
    app_flask.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

def run_auto_sell():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(auto_sell_loop())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        logger.info("Auto-sell loop closed")

if __name__ == "__main__":
    logger.info("Starting bot")
    required_envs = [
        "API_ID", "API_HASH", "SESSION_STRING", "BOT_TOKEN",
        "SOLANA_PRIVATE_KEY", "SOLANA_RPC", "JUPITER_API", "USDC_MINT"
    ]
    for env in required_envs:
        if not os.environ.get(env):
            logger.error(f"Missing env: {env}")
            exit(1)

    # Start Telethon client
    tele_client.start()
    logger.info("Telethon client started")

    # Start Flask in a thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("Flask server started")

    # Start auto-sell loop in a thread
    auto_sell_thread = threading.Thread(target=run_auto_sell, daemon=True)
    auto_sell_thread.start()
    logger.info("Auto-sell loop started")

    # Run Telegram bot
    app.run_polling()
    logger.info("Bot shutdown complete")
