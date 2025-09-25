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
import struct
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
from decimal import Decimal, getcontext
getcontext().prec = 28  # high precision for token math
MAX_U64 = 2**64 - 1

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
CHAT_ID = os.environ.get("CHAT_ID")
FALLBACK_CHAT_ID = os.environ.get("FALLBACK_CHAT_ID")  # Optional fallback chat ID

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
last_command_chat_id = None  # Store the chat ID from the latest command

def load_trades():
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_trades(trades):
    try:
        with trades_lock:
            with open(TRADES_FILE, "w") as f:
                json.dump(trades, f, indent=2)
            logger.info("Trades successfully saved to %s", TRADES_FILE)
    except Exception as e:
        logger.error(f"Failed to save trades: {e}")

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

def get_token_info(mint_str: str):
    try:
        mint = Pubkey.from_string(mint_str)
        token = Token(sol_client, mint, TOKEN_PROGRAM_ID, payer)
        mint_info = token.get_mint_info()
        decimals = mint_info.decimals
        total_supply = mint_info.supply / (10 ** decimals)
        logger.info(f"Fetched mint info for {mint_str}: decimals={decimals}, total_supply={total_supply}")

        # Attempt to fetch metadata
        metadata_program_id = Pubkey.from_string("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s")
        metadata_pda, _ = Pubkey.find_program_address(
            [b"metadata", bytes(metadata_program_id), bytes(mint)],
            metadata_program_id
        )
        account_info = sol_client.get_account_info(metadata_pda)
        if account_info.value is None:
            logger.warning(f"No metadata found for {mint_str}")
            return {"name": "Unknown", "decimals": decimals, "total_supply": total_supply}

        data = account_info.value.data
        if len(data) < 69:
            logger.warning(f"Metadata for {mint_str} is too short: {len(data)} bytes")
            return {"name": "Unknown", "decimals": decimals, "total_supply": total_supply}

        try:
            name_len = struct.unpack("<I", data[65:69])[0]
            if 69 + name_len > len(data):
                logger.warning(f"Invalid name length for {mint_str}: name_len={name_len}, data_len={len(data)}")
                return {"name": "Unknown", "decimals": decimals, "total_supply": total_supply}
            name = data[69:69 + name_len].decode("utf-8").rstrip("\x00")
            logger.info(f"Fetched token name for {mint_str}: {name}")
            return {"name": name, "decimals": decimals, "total_supply": total_supply}
        except Exception as e:
            logger.error(f"Failed to parse metadata for {mint_str}: {e}")
            return {"name": "Unknown", "decimals": decimals, "total_supply": total_supply}
    except Exception as e:
        logger.error(f"Failed to get token info for {mint_str}: {e}")
        return {"name": "Unknown", "decimals": 6, "total_supply": 0}

# ------------------ TELEGRAM BOT ------------------ #
app = ApplicationBuilder().token(BOT_TOKEN).build()
logger.info("Telegram bot application initialized")

# ------------------ TELETHON CLIENT ------------------ #
tele_client = TelegramClient(StringSession(SESSION_STRING), API_ID, API_HASH)
logger.info("Telethon client initialized")

# ------------------ JUPITER FUNCTIONS ------------------ #

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_swap_quote(input_mint, output_mint, amount, restrict_intermediates=False, only_direct_routes=False):
    logger.info(f"Getting swap quote: input={input_mint}, output={output_mint}, amount={amount}, slippage={SLIPPAGE_BPS}")
    decimals_in = get_token_decimals(input_mint)
    raw_amount = int(amount * (10 ** decimals_in))

    url = f"{JUPITER_API}/quote"
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": raw_amount,
        "slippageBps": SLIPPAGE_BPS,
        "onlyDirectRoutes": str(only_direct_routes).lower(),  # Convert boolean to lowercase string
    }
    if restrict_intermediates:
        params["restrictIntermediateTokens"] = str(restrict_intermediates).lower()  # Ensure boolean is string

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

def execute_swap_route(route, output_mint, *, attempt_reduce_amount=False):
    """
    Executes a Jupiter swap given a quote `route`.
    - Automatically handles signing and sending.
    - Retries with safer route / smaller amount if Anchor error 6024 (0x1788) occurs.
    - Keeps wrapAndUnwrapSol=False per your request.
    """
    try:
        logger.info("Getting swap transaction from Jupiter API")

        swap_url = f"{JUPITER_API}/swap"
        swap_payload = {
            "quoteResponse": route,
            "userPublicKey": str(payer.pubkey()),
            "wrapAndUnwrapSol": False,             # <-- per your request
            "dynamicComputeUnitLimit": True,
            "prioritizationFeeLamports": 0
        }

        swap_resp = requests.post(swap_url, json=swap_payload, timeout=20)
        try:
            swap_resp.raise_for_status()
        except Exception:
            logger.error("Jupiter /swap returned failure: %s", swap_resp.text)
            return None

        response_json = swap_resp.json()
        logger.debug("Swap response JSON: %s", response_json)

        tx_base64 = response_json.get("swapTransaction") or response_json.get("swap_transaction")
        if not tx_base64:
            logger.error("No swapTransaction in Jupiter /swap response: %s", response_json)
            return None

        tx_bytes = base64.b64decode(tx_base64)
        raw_tx = VersionedTransaction.from_bytes(tx_bytes)

        # sign
        to_sign = to_bytes_versioned(raw_tx.message)
        try:
            signature = payer.sign_message(to_sign)
            signed_tx = VersionedTransaction.populate(raw_tx.message, [signature])
            serialized = bytes(signed_tx)
        except Exception as sign_exc:
            logger.warning("Signing failed: %s", sign_exc)
            if hasattr(raw_tx, "create_signatures"):
                sigs = raw_tx.create_signatures([payer])
                raw_tx.set_signatures(sigs)
                serialized = bytes(raw_tx)
            else:
                logger.exception("All signing fallbacks failed")
                return None

        # send transaction
        try:
            opts = TxOpts(skip_preflight=False, preflight_commitment="confirmed")
            resp = sol_client.send_raw_transaction(serialized, opts=opts)
            sig = resp.value
            logger.info("Transaction signature: %s", sig)
            sol_client.confirm_transaction(sig, commitment="confirmed")
            logger.info("Transaction confirmed: https://solscan.io/tx/%s", sig)
            return resp
        except Exception as e:
            err_text = str(e)
            logger.error("First send attempt failed: %s", err_text)

            # if 6024/0x1788, retry with safer route
            if "6024" in err_text or "0x1788" in err_text:
                logger.warning("Detected custom program error 6024 (0x1788). Trying safer fallbacks: restrict routes / reduce amount.")
                try:
                    input_mint = route.get("inputMint") or route.get("inputMint")
                    output_mint = route.get("outputMint") or output_mint
                    in_amount_raw = int(route.get("inAmount") or route.get("in_amount") or 0)
                    decimals = get_token_decimals(input_mint)
                    human_amount = Decimal(in_amount_raw) / (Decimal(10) ** decimals)

                    # 1) try direct/simpler route
                    safe_quote = get_swap_quote(
                        input_mint, output_mint, float(human_amount),
                        restrict_intermediates=True, only_direct_routes=True
                    )
                    if safe_quote:
                        logger.info("Found safe quote (direct). Retrying swap via /swap with simpler route")
                        return execute_swap_route(safe_quote, output_mint, attempt_reduce_amount=attempt_reduce_amount)

                    # 2) try selling slightly less (99.9%)
                    if not attempt_reduce_amount:
                        reduced_amount = float(human_amount * Decimal('0.999'))
                        logger.info("Retrying with slightly reduced amount (%s -> %s)", human_amount, reduced_amount)
                        reduced_quote = get_swap_quote(
                            input_mint, output_mint, reduced_amount,
                            restrict_intermediates=True
                        )
                        if reduced_quote:
                            return execute_swap_route(reduced_quote, output_mint, attempt_reduce_amount=True)
                except Exception as inner_exc:
                    logger.exception("Fallback attempts failed: %s", inner_exc)

            return None

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

# ------------------ HELPER FUNCTIONS ------------------ #

def get_current_info(token):
    current_price = fetch_token_price(token)
    info = trades.get(token, {})
    name = info.get("name", "Unknown")
    total_supply = info.get("total_supply", 0)
    buy_price = info.get("buy_price", 0)
    amount = info.get("amount", 0)
    if current_price is None:
        profit_percent = 0
        profit_usd = 0
        market_cap = "N/A"
        current_price = "N/A"
    else:
        profit_percent = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
        profit_usd = (current_price - buy_price) * amount
        market_cap = total_supply * current_price if total_supply > 0 else "N/A"
    return name, current_price, profit_percent, profit_usd, market_cap

def get_status_content():
    msg = f"Auto-buy: {AUTO_BUY}\nTrade amount: {TRADE_AMOUNT_USDC} USDC\nSlippage: {SLIPPAGE_BPS/100}%\nChannels: {CHANNELS}\nPreset Sell Targets: {PRESET_SELL_TARGETS}\nPreset Stop-Loss: {[f'{x}%' for x in PRESET_STOP_LOSS]}\n\nTracked Tokens:\n"
    for token, info in trades.items():
        short_addr = f"{token[:6]}...{token[-6:]}"
        name, current_price, profit_percent, profit_usd, market_cap = get_current_info(token)
        msg += f"{name} ({short_addr}): {info['amount']} tokens bought at {info['buy_price']}, P/L: {profit_percent:.2f}% ({profit_usd:.2f} USDC), MC: {market_cap if isinstance(market_cap, str) else f'{market_cap:.2f}'}\n"
    keyboard = [
        [InlineKeyboardButton("Toggle Auto-Buy", callback_data="togglebuy"), InlineKeyboardButton("Set Amount", callback_data="setamount")],
        [InlineKeyboardButton("Add Channel", callback_data="addchannel"), InlineKeyboardButton("Remove Channel", callback_data="removechannel")],
        [InlineKeyboardButton("View Trades", callback_data="viewtrades"), InlineKeyboardButton("Change Wallet", callback_data="setwallet")],
        [InlineKeyboardButton("Set Targets", callback_data="settargets"), InlineKeyboardButton("Set Stop-Loss", callback_data="setstoploss")],
        [InlineKeyboardButton("Sell Token", callback_data="sell"), InlineKeyboardButton("Set Slippage", callback_data="setslippage")],
        [InlineKeyboardButton("Set Preset Targets", callback_data="setpresettargets"), InlineKeyboardButton("Set Preset Stop-Loss", callback_data="setpresetstoploss")],
        [InlineKeyboardButton("Refresh", callback_data="refreshstatus")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return msg, reply_markup

def get_viewtrades_msg():
    msg = "Tracked Tokens:\n"
    for token, info in trades.items():
        name, current_price, _, _, market_cap = get_current_info(token)
        msg += f"Token: {name} ({token})\nAmount: {info['amount']}\nBuy Price: {info['buy_price']}\nCurrent Price: {current_price}\nMarket Cap: {market_cap if isinstance(market_cap, str) else f'{market_cap:.2f}'}\n\n"
    return msg or "No trades yet"

# ------------------ NOTIFICATION FUNCTION ------------------ #

async def send_notification(message, token_mint, action="buy"):
    global last_command_chat_id
    chat_ids = []
    if last_command_chat_id:
        chat_ids.append(last_command_chat_id)
    if CHAT_ID and CHAT_ID != last_command_chat_id:
        chat_ids.append(CHAT_ID)
    if FALLBACK_CHAT_ID and FALLBACK_CHAT_ID not in chat_ids:
        chat_ids.append(FALLBACK_CHAT_ID)

    for chat_id in chat_ids:
        try:
            await app.bot.send_message(chat_id=int(chat_id), text=message)
            logger.info(f"Sent {action} notification for {token_mint} to chat {chat_id}")
            return True  # Stop after first successful send
        except Exception as e:
            logger.error(f"Failed to send {action} notification for {token_mint} to chat {chat_id}: {e}. "
                         f"Ensure the bot is added to the chat with send message permissions. "
                         f"Use @userinfobot or @getidsbot to verify chat ID. "
                         f"Current CHAT_ID={CHAT_ID}, BOT_TOKEN starts with {BOT_TOKEN[:10]}...")
    logger.warning(f"No valid chat IDs for {action} notification of {token_mint}")
    return False

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
        if not resp:
            logger.error(f"Swap execution failed for {token_mint}")
            return
        
        logger.info(f"Swap successful for {token_mint}, fetching price")
        buy_price = fetch_token_price(token_mint)
        if buy_price is None:
            logger.error(f"Failed to fetch buy price for {token_mint}. Proceeding with default values.")
            buy_price = 0

        out_amount_base = int(route["outAmount"])
        token_decimals = get_token_decimals(token_mint)
        out_amount_human = out_amount_base / (10 ** token_decimals)

        token_info = get_token_info(token_mint)
        name = token_info["name"] if token_info else "Unknown"
        total_supply = token_info["total_supply"] if token_info else 0
        market_cap = total_supply * buy_price if buy_price and total_supply > 0 else "N/A"

        logger.info(f"PRESET_STOP_LOSS: {PRESET_STOP_LOSS}")
        stop_loss = PRESET_STOP_LOSS[2] if len(PRESET_STOP_LOSS) > 2 else 30

        trades[token_mint] = {
            "buy_price": buy_price,
            "amount": out_amount_human,
            "targets": PRESET_SELL_TARGETS.copy(),
            "stop_loss": stop_loss,
            "name": name,
            "total_supply": total_supply,
        }
        logger.info(f"Added {token_mint} to trades: {trades[token_mint]}")
        save_trades(trades)
        logger.info(
            f"Bought {out_amount_human} of {token_mint} ({name}) at {buy_price} USDC "
            f"with targets {trades[token_mint]['targets']} and stop-loss {trades[token_mint]['stop_loss']}%"
        )

        # Send notification
        notif_msg = f"Bought {out_amount_human} {name} ({token_mint}) at {buy_price} USDC each, market cap: {market_cap if isinstance(market_cap, str) else f'{market_cap:,.2f}'}"
        await send_notification(notif_msg, token_mint, action="buy")

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

            name = info.get("name", "Unknown")
            total_supply = info.get("total_supply", 0)
            market_cap = total_supply * current_price if total_supply > 0 else "N/A"
            amount = info["amount"]
            buy_price = info["buy_price"]
            buy_cost = buy_price * amount

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
                        effective_sell_price = out_amount_human / amount if amount > 0 else 0
                        profit_usd = out_amount_human - buy_cost
                        profit_percent = (profit_usd / buy_cost * 100) if buy_cost > 0 else 0
                        logger.info(f"Sold {info['amount']} of {token} for {out_amount_human} USDC at {current_price} (target {target}x)")

                        # Send notification
                        notif_msg = f"Sold {amount} {name} ({token}) for {out_amount_human} USDC (effective price {effective_sell_price}), profit {profit_percent:+.2f}% ({profit_usd:+.2f} USDC), market cap at sell: {market_cap if isinstance(market_cap, str) else f'{market_cap:,.2f}'} (target {target}x)"
                        await send_notification(notif_msg, token, action="sell")
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
                    effective_sell_price = out_amount_human / amount if amount > 0 else 0
                    profit_usd = out_amount_human - buy_cost
                    profit_percent = (profit_usd / buy_cost * 100) if buy_cost > 0 else 0
                    logger.info(f"Sold {info['amount']} of {token} for {out_amount_human} USDC at {current_price} (stop-loss {info['stop_loss']}%)")

                    # Send notification
                    notif_msg = f"Sold {amount} {name} ({token}) for {out_amount_human} USDC (effective price {effective_sell_price}), profit {profit_percent:+.2f}% ({profit_usd:+.2f} USDC), market cap at sell: {market_cap if isinstance(market_cap, str) else f'{market_cap:,.2f}'} (stop-loss {info['stop_loss']}%)"
                    await send_notification(notif_msg, token, action="sell")
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

# ------------------ TELEGRAM COMMANDS ------------------ #

async def update_last_command_chat_id(update: Update):
    global last_command_chat_id
    if update.message and update.message.chat_id:
        last_command_chat_id = str(update.message.chat_id)
        logger.info(f"Updated last_command_chat_id to {last_command_chat_id}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /status command")
    await update_last_command_chat_id(update)
    msg, reply_markup = get_status_content()
    await update.message.reply_text(msg, reply_markup=reply_markup)

async def viewtrades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /viewtrades command")
    await update_last_command_chat_id(update)
    msg = get_viewtrades_msg()
    await update.message.reply_text(msg)

async def setamount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setamount command")
    await update_last_command_chat_id(update)
    global TRADE_AMOUNT_USDC
    try:
        TRADE_AMOUNT_USDC = float(context.args[0])
        await update.message.reply_text(f"Trade amount updated to {TRADE_AMOUNT_USDC} USDC")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /setamount <amount>")

async def settargets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /settargets command")
    await update_last_command_chat_id(update)
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
    await update_last_command_chat_id(update)
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
    await update_last_command_chat_id(update)
    global AUTO_BUY
    AUTO_BUY = not AUTO_BUY
    await update.message.reply_text(f"Auto-buy set to {AUTO_BUY}")

async def addchannel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /addchannel command")
    await update_last_command_chat_id(update)
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
    await update_last_command_chat_id(update)
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
    await update_last_command_chat_id(update)
    try:
        token = context.args[0]
        if token in trades:
            info = trades[token]
            name = info.get("name", "Unknown")
            total_supply = info.get("total_supply", 0)
            amount = info["amount"]
            buy_price = info["buy_price"]
            buy_cost = buy_price * amount
            current_price = fetch_token_price(token)
            market_cap = total_supply * current_price if current_price and total_supply > 0 else "N/A"

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
                effective_sell_price = out_amount_human / amount if amount > 0 else 0
                profit_usd = out_amount_human - buy_cost
                profit_percent = (profit_usd / buy_cost * 100) if buy_cost > 0 else 0
                amount_sold = trades[token]['amount']
                trades.pop(token)
                save_trades(trades)
                await update.message.reply_text(f"Manually sold {amount_sold} of {token} for {out_amount_human} USDC")

                # Send notification
                notif_msg = f"Manually sold {amount} {name} ({token}) for {out_amount_human} USDC (effective price {effective_sell_price}), profit {profit_percent:+.2f}% ({profit_usd:+.2f} USDC), market cap at sell: {market_cap if isinstance(market_cap, str) else f'{market_cap:,.2f}'}"
                await send_notification(notif_msg, token, action="sell")
            else:
                await update.message.reply_text(f"Failed to sell {token}: Swap rejected or failed")
        else:
            await update.message.reply_text(f"{token} not tracked")
    except Exception as e:
        await update.message.reply_text(f"Usage: /sell <token> | Error: {str(e)}")

async def setwallet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setwallet command")
    await update_last_command_chat_id(update)
    global payer
    try:
        new_key = context.args[0]
        payer = Keypair.from_bytes(bytes.fromhex(new_key))
        await update.message.reply_text("Wallet updated successfully")
    except Exception as e:
        await update.message.reply_text(f"Failed to update wallet: {e}")

async def setslippage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Received /setslippage command")
    await update_last_command_chat_id(update)
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
    await update_last_command_chat_id(update)
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
    await update_last_command_chat_id(update)
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
    await update_last_command_chat_id(update)
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
        msg = get_viewtrades_msg()
        await query.message.reply_text(msg)
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
    elif query.data == "refreshstatus":
        msg, reply_markup = get_status_content()
        await query.edit_message_text(msg, reply_markup=reply_markup)

app.add_handler(CommandHandler("status", status))
app.add_handler(CommandHandler("viewtrades", viewtrades))
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

# ------------------ CHAT_ID VALIDATION ------------------ #
async def validate_chat_id():
    chat_ids = [CHAT_ID]
    if FALLBACK_CHAT_ID and FALLBACK_CHAT_ID != CHAT_ID:
        chat_ids.append(FALLBACK_CHAT_ID)
    for chat_id in chat_ids:
        try:
            await app.bot.send_message(chat_id=int(chat_id), text="Bot initialized successfully")
            logger.info(f"Successfully validated chat ID: {chat_id}")
        except Exception as e:
            logger.error(f"Failed to validate chat ID {chat_id}: {e}. "
                         f"Ensure the bot is added to the chat with send message permissions. "
                         f"Use @userinfobot or @getidsbot to verify chat ID. "
                         f"Current CHAT_ID={CHAT_ID}, BOT_TOKEN starts with {BOT_TOKEN[:10]}...")

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
    try:
        CHAT_ID = int(CHAT_ID)
        logger.info(f"CHAT_ID set to {CHAT_ID}")
    except ValueError:
        logger.error(f"Invalid CHAT_ID: {CHAT_ID}. Must be a valid integer.")
        exit(1)

    tele_client.start()
    logger.info("Telethon client started")
    
    # Validate chat IDs at startup
    loop = asyncio.get_event_loop()
    loop.run_until_complete(validate_chat_id())
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("Flask server started")
    
    auto_sell_thread = threading.Thread(target=run_auto_sell, daemon=True)
    auto_sell_thread.start()
    logger.info("Auto-sell loop started")
    
    app.run_polling()
    logger.info("Bot shutdown complete")
