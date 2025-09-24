# bot.py
from solders.transaction import VersionedTransaction
from solders.message import to_bytes_versioned
from solders.signature import Signature
from solana.rpc.types import TxOpts
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
# --- IMPORTANT: Using v6 API ---
JUPITER_API = os.environ.get("JUPITER_API", "https://quote-api.jup.ag/v6")
USDC_MINT = os.environ.get("USDC_MINT", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
CHANNELS = os.environ.get("CHANNELS", "").split(",")
CHAT_ID = int(os.environ.get("CHAT_ID"))  # Your Telegram user ID for notifications

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

def get_token_ticker(mint: str) -> str:
    try:
        metadata_program = Pubkey.from_string("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s")
        mint_pubkey = Pubkey.from_string(mint)
        metadata_pda = Pubkey.find_program_address([b"metadata", metadata_program.to_bytes(), mint_pubkey.to_bytes()], metadata_program)[0]
        account_info = sol_client.get_account_info(metadata_pda)
        if account_info.value:
            data = account_info.value.data
            # Name length at 32-36, name at 36+
            name_length = int.from_bytes(data[32:36], "little")
            symbol_offset = 36 + name_length + 4  # Skip name and symbol length
            symbol_length = int.from_bytes(data[36 + name_length:36 + name_length + 4], "little")
            symbol = data[symbol_offset:symbol_offset + symbol_length].decode('utf-8').strip('\x00')
            return symbol or "UNK"
        return "UNK"
    except Exception as e:
        logger.error(f"Metadata fetch failed for mint {mint}")
        return "UNK"

def get_market_cap(token_mint: str, price: float) -> str:
    try:
        mint_pubkey = Pubkey.from_string(token_mint)
        token = Token(sol_client, mint_pubkey, TOKEN_PROGRAM_ID, payer)
        supply = token.get_mint_info().supply
        decimals = get_token_decimals(token_mint)
        total_tokens = supply / (10 ** decimals)
        mc = total_tokens * price
        return f"${mc:,.2f}"
    except Exception as e:
        logger.error(f"Failed to calculate market cap for {token_mint}: {e}")
        return "N/A"

def shorten_mint(mint: str) -> str:
    return f"{mint[:8]}...{mint[-8:]}"

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
        # --- IMPROVEMENT: Use lowercase string "false" for v6 API ---
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
    """
    route = the quote response you passed to /swap (Jupiter's v6)
    returns solana-py response (or None on failure)
    """
    try:
        logger.info("Getting swap transaction from Jupiter API")
        swap_url = f"{JUPITER_API}/swap"

        # ⬇️ This is the only part changed: add the low-fee params
        swap_payload = {
            "quoteResponse": route,
            "userPublicKey": str(payer.pubkey()),
            "wrapAndUnwrapSol": False,        # cheaper if not swapping SOL
            "dynamicComputeUnitLimit": True,  # let Jupiter shrink compute usage
            "prioritizationFeeLamports": 0    # tell Jupiter no priority tip
        }

        swap_resp = requests.post(swap_url, json=swap_payload, timeout=20)
        swap_resp.raise_for_status()
        response_json = swap_resp.json()

        # find the base64 transaction in the response
        tx_base64 = response_json.get("swapTransaction") or response_json.get("swap_transaction")
        if not tx_base64:
            logger.error("No swapTransaction in Jupiter /swap response: %s", response_json)
            return None

        tx_bytes = base64.b64decode(tx_base64)
        raw_tx = VersionedTransaction.from_bytes(tx_bytes)

        # Sign the versioned message bytes
        try:
            to_sign = to_bytes_versioned(raw_tx.message)
            signature = payer.sign_message(to_sign)
            signed_tx = VersionedTransaction.populate(raw_tx.message, [signature])
            serialized = bytes(signed_tx)
        except Exception as sign_exc:
            logger.warning("Primary signing method failed (%s), trying alternate signing paths", sign_exc)
            if hasattr(raw_tx, "create_signatures"):
                sigs = raw_tx.create_signatures([payer])
                raw_tx.set_signatures(sigs)
                serialized = bytes(raw_tx)
            else:
                try:
                    resp = sol_client.send_transaction(raw_tx, payer,
                        opts=TxOpts(skip_preflight=False, preflight_commitment="confirmed"))
                    logger.info("Sent via send_transaction fallback, resp: %s", resp)
                    return resp
                except Exception as e:
                    logger.error("All signing fallbacks failed: %s", e)
                    return None

        # Send the already-signed, serialized transaction
        opts = TxOpts(skip_preflight=False, preflight_commitment="confirmed")
        resp = sol_client.send_raw_transaction(serialized, opts=opts)
        sig = resp.value
        logger.info("Transaction signature: %s", sig)
        sol_client.confirm_transaction(sig, commitment="confirmed")
        logger.info("Transaction confirmed: https://solscan.io/tx/%s", sig)
        return resp

    except requests.exceptions.RequestException as re:
        logger.error("HTTP error talking to Jupiter swap: %s", re)
        return None
    except Exception as e:
        logger.exception("Swap execution failed: %s", e)
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
        
        # The quote object is the route itself in v6
        route = quote
        
        logger.info(f"Got swap quote, executing swap for {token_mint}")
        resp = execute_swap_route(route, token_mint)
        if resp:
            logger.info(f"Swap successful for {token_mint}, fetching price")
            buy_price = fetch_token_price(token_mint)
            
            out_amount_base = int(route["outAmount"])
            token_decimals = get_token_decimals(token_mint)
            out_amount_human = out_amount_base / (10 ** token_decimals)

            ticker = get_token_ticker(token_mint)
            mc = get_market_cap(token_mint, buy_price)
            sig = resp.value
            await app.bot.send_message(chat_id=CHAT_ID, text=f"🟢 Bought {out_amount_human} {ticker} ✅\n🔴 Market Cap {mc}\n💲 Amount Spent {TRADE_AMOUNT_USDC}\n🔗 https://solscan.io/tx/{sig}")

            trades[token_mint] = {
                "buy_price": buy_price,
                "amount": out_amount_human,
                "targets": PRESET_SELL_TARGETS.copy(),
                "stop_loss": PRESET_STOP_LOSS[2],
                "symbol": ticker,
                "current_price": buy_price,
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

            # Update cache
            trades[token]["current_price"] = current_price

            # Check take-profit targets
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
                        percent_change = ((current_price - info["buy_price"]) / info["buy_price"]) * 100
                        usdc_change = (current_price - info["buy_price"]) * info["amount"]
                        change_str = f"{percent_change:+.2f}% ({usdc_change:+.2f}$)"
                        ticker = info.get("symbol", "UNK")
                        mc = get_market_cap(token, current_price)
                        sig = resp.value
                        await app.bot.send_message(chat_id=CHAT_ID, text=f"🔴 Sold {info['amount']} {ticker}\n⚪ Market Cap {mc}\n💲 Amount gotten {out_amount_human}\n📈 Profit/Loss: {change_str}\n🔗 https://solscan.io/tx/{sig}")

                        logger.info(f"Sold {info['amount']} of {token} for {out_amount_human} USDC at {current_price} (target {target}x)")
                        to_remove.append(token)
                        break
            
            if token in to_remove: continue

            # Check stop-loss
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
                    percent_change = ((current_price - info["buy_price"]) / info["buy_price"]) * 100
                    usdc_change = (current_price - info["buy_price"]) * info["amount"]
                    change_str = f"{percent_change:+.2f}% ({usdc_change:+.2f}$)"
                    ticker = info.get("symbol", "UNK")
                    mc = get_market_cap(token, current_price)
                    sig = resp.value
                    await app.bot.send_message(chat_id=CHAT_ID, text=f"🔴 Sold {info['amount']} {ticker}\n⚪ Market Cap {mc}\n💲 Amount gotten {out_amount_human}\n📈 Profit/Loss: {change_str}\n🔗 https://solscan.io/tx/{sig}")

                    logger.info(f"Sold {info['amount']} of {token} for {out_amount_human} USDC at {current_price} (stop-loss {info['stop_loss']}%)")
                    to_remove.append(token)
        except Exception as e:
            logger.error(f"Auto-sell error for {token}: {e}")
    for token in to_remove:
        if token in trades: trades.pop(token)
    save_trades(trades)

async def update_prices():
    for token in trades.keys():
        current_price = fetch_token_price(token)
        if current_price is not None:
            trades[token]["current_price"] = current_price
    save_trades(trades)

async def auto_sell_loop():
    logger.info("Starting auto-sell loop")
    while True:
        await check_auto_sell()
        await update_prices()
        await asyncio.sleep(60)

# ------------------ TELEGRAM COMMANDS ------------------ #
def generate_status_msg():
    msg = f"Auto-buy: {AUTO_BUY}\nTrade amount: {TRADE_AMOUNT_USDC} USDC\nSlippage: {SLIPPAGE_BPS/100}%\nChannels: {CHANNELS}\nPreset Sell Targets: {PRESET_SELL_TARGETS}\nPreset Stop-Loss: {[f'{x}%' for x in PRESET_STOP_LOSS]}\n\nTracked Tokens:\n"
    for token, info in trades.items():
        ticker = info.get("symbol", "UNK")
        shortened_ca = shorten_mint(token)
        current_price = info.get("current_price")
        if current_price is not None:
            percent_change = ((current_price - info["buy_price"]) / info["buy_price"]) * 100
            usdc_change = (current_price - info["buy_price"]) * info["amount"]
            emoji = "🟢" if percent_change > 0 else "🔴"
            change_str = f" ({percent_change:+.2f}%) ({usdc_change:+.2f}$)"
        else:
            emoji = ""
            change_str = " (N/A) (N/A)"
        msg += f"⚪ {ticker} ({shortened_ca})\n{emoji}{change_str} Buy {info['amount']} tokens at {info['buy_price']}, Targets {info['targets']}, Stop-loss {info['stop_loss']}%\n"
    return msg

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /status command")
    msg = generate_status_msg()
    keyboard = [
        [InlineKeyboardButton("Toggle Auto-Buy", callback_data="togglebuy"), InlineKeyboardButton("Set Amount", callback_data="setamount")],
        [InlineKeyboardButton("Add Channel", callback_data="addchannel"), InlineKeyboardButton("Remove Channel", callback_data="removechannel")],
        [InlineKeyboardButton("View Trades", callback_data="viewtrades"), InlineKeyboardButton("Change Wallet", callback_data="setwallet")],
        [InlineKeyboardButton("Set Targets", callback_data="settargets"), InlineKeyboardButton("Set Stop-Loss", callback_data="setstoploss")],
        [InlineKeyboardButton("Sell Token", callback_data="sell"), InlineKeyboardButton("Set Slippage", callback_data="setslippage")],
        [InlineKeyboardButton("Set Preset Targets", callback_data="setpresettargets"), InlineKeyboardButton("Set Preset Stop-Loss", callback_data="setpresetstoploss")],
        [InlineKeyboardButton("Refresh", callback_data="refresh")],
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
            info = trades[token]
            quote = get_swap_quote(token, USDC_MINT, info["amount"])
            if not quote:
                await update.message.reply_text(f"Could not get a quote to sell {token}")
                return
            route = quote
            resp = execute_swap_route(route, USDC_MINT)
            if resp:
                out_amount_base = int(route["outAmount"])
                usdc_decimals = get_token_decimals(USDC_MINT)
                out_amount_human = out_amount_base / (10 ** usdc_decimals)
                current_price = fetch_token_price(token)
                percent_change = ((current_price - info["buy_price"]) / info["buy_price"]) * 100 if current_price else 0
                usdc_change = (current_price - info["buy_price"]) * info["amount"] if current_price else 0
                change_str = f"{percent_change:+.2f}% ({usdc_change:+.2f}$)"
                ticker = info.get("symbol", "UNK")
                mc = get_market_cap(token, current_price) if current_price else "N/A"
                sig = resp.value
                await app.bot.send_message(chat_id=CHAT_ID, text=f"🔴 Sold {info['amount']} {ticker}\n⚪ Market Cap {mc}\n💲 Amount gotten {out_amount_human}\n📈 Profit/Loss: {change_str}\n🔗 https://solscan.io/tx/{sig}")

                amount_sold = info['amount']
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
    elif query.data == "setamount":
        await query.message.reply_text("Send /setamount <amount> to set trade amount")
    elif query.data == "addchannel":
        await query.message.reply_text("Send /addchannel @channelname to add a channel")
    elif query.data == "removechannel":
        await query.message.reply_text("Send /removechannel @channelname to remove a channel")
    elif query.data == "viewtrades":
        msg = "Tracked Tokens:\n"
        for token, info in trades.items():
            ticker = info.get("symbol", "UNK")
            mc = get_market_cap(token, info.get("current_price", info["buy_price"]))
            msg += f"{ticker} ({token})\nAmount: {info['amount']} tokens\nBuy Price: {info['buy_price']} USDC\nMarket Cap: {mc}\n\n"
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
        await query.message.reply_text("Send /setslippage <percentage> to set slippage (0.1 to 50)")
    elif query.data == "setpresettargets":
        await query.message.reply_text("Send /setpresettargets <x1,x2,...> to set preset sell targets")
    elif query.data == "setpresetstoploss":
        await query.message.reply_text("Send /setpresetstoploss <x1,x2,...> to set preset stop-loss")
    elif query.data == "refresh":
        msg = generate_status_msg()
        keyboard = query.message.reply_markup.inline_keyboard
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text=msg, reply_markup=reply_markup)

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
                    await buy_token(token)
                else:
                    logger.debug(f"Filtered out invalid potential token: {token}")

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
    required_envs = ["API_ID", "API_HASH", "SESSION_STRING", "BOT_TOKEN", "SOLANA_PRIVATE_KEY", "CHAT_ID"]
    for env in required_envs:
        if not os.environ.get(env):
            logger.error(f"Missing required environment variable: {env}")
            exit(1)

    tele_client.start()
    logger.info("Telethon client started")
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("Flask server started")
    
    auto_sell_thread = threading.Thread(target=run_auto_sell, daemon=True)
    auto_sell_thread.start()
    logger.info("Auto-sell loop started")
    
    app.run_polling()
    logger.info("Bot shutdown complete")
