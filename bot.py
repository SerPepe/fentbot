import torch
from torch import autocast
from diffusers import DiffusionPipeline
from PIL import Image

import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from io import BytesIO
import random

load_dotenv()
TG_TOKEN = os.getenv('TG_TOKEN')
MODEL_DATA = os.getenv('MODEL_DATA', 'thegovind/pills1testmodel')
LOW_VRAM_MODE = (os.getenv('LOW_VRAM', 'true').lower() == 'true')
USE_AUTH_TOKEN = (os.getenv('USE_AUTH_TOKEN', 'true').lower() == 'true')
HEIGHT = int(os.getenv('HEIGHT', '512'))
WIDTH = int(os.getenv('WIDTH', '512'))
NUM_INFERENCE_STEPS = int(os.getenv('NUM_INFERENCE_STEPS', '50'))
STRENGTH = float(os.getenv('STRENGTH', '0.75'))
GUIDANCE_SCALE = float(os.getenv('GUIDANCE_SCALE', '7.5'))

torch_dtype = torch.float32 if LOW_VRAM_MODE else None

# Load the pipeline
pipe = DiffusionPipeline.from_pretrained(MODEL_DATA, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def image_to_bytes(image):
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)
    return bio

def get_try_again_markup():
    keyboard = [[InlineKeyboardButton("Try again", callback_data="TRYAGAIN"), InlineKeyboardButton("Variations", callback_data="VARIATIONS")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup

def generate_image(prompt, seed=None, height=HEIGHT, width=WIDTH, num_inference_steps=NUM_INFERENCE_STEPS, strength=STRENGTH, guidance_scale=GUIDANCE_SCALE):
    seed = seed if seed is not None else random.randint(1, 10000)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    with autocast("cuda"):
        image = pipe(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps, generator=generator)["sample"][0]
    return image, seed

async def generate_and_send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    im, seed = generate_image(prompt=update.message.text)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.effective_user.id, image_to_bytes(im), caption=f'"{update.message.text}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    prompt = query.message.reply_to_message.text
    im, seed = generate_image(prompt=prompt)
    await context.bot.send_photo(query.message.chat_id, image_to_bytes(im), caption=f'"{prompt}" (Seed: {seed})', reply_markup=get_try_again_markup())

app = ApplicationBuilder().token(TG_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_and_send_photo))
app.add_handler(CallbackQueryHandler(button))
app.run_polling()
