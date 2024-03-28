import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters
from io import BytesIO
import random

load_dotenv()
TG_TOKEN = os.getenv('TG_TOKEN')
MODEL_DATA = os.getenv('MODEL_DATA', 'runwayml/stable-diffusion-v1-5')
LOW_VRAM_MODE = (os.getenv('LOW_VRAM', 'true').lower() == 'true')
USE_AUTH_TOKEN = (os.getenv('USE_AUTH_TOKEN', 'true').lower() == 'true')
SAFETY_CHECKER = (os.getenv('SAFETY_CHECKER', 'true').lower() == 'true')
HEIGHT = int(os.getenv('HEIGHT', '512'))
WIDTH = int(os.getenv('WIDTH', '512'))
NUM_INFERENCE_STEPS = int(os.getenv('NUM_INFERENCE_STEPS', '50'))
STRENGTH = float(os.getenv('STRENGTH', '0.75'))
GUIDANCE_SCALE = float(os.getenv('GUIDANCE_SCALE', '7.5'))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipelines
pipe = StableDiffusionPipeline.from_pretrained(MODEL_DATA, use_auth_token=USE_AUTH_TOKEN).to(device)
img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_DATA, use_auth_token=USE_AUTH_TOKEN).to(device)

# Disable safety checker if desired
if not SAFETY_CHECKER:
    def dummy_checker(images, **kwargs):
        # Return a list of False, one for each image
        return images, [False] * len(images)
    pipe.safety_checker = dummy_checker
    img2imgPipe.safety_checker = dummy_checker
    
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

def generate_image(prompt, seed=None, height=HEIGHT, width=WIDTH, num_inference_steps=NUM_INFERENCE_STEPS, strength=STRENGTH, guidance_scale=GUIDANCE_SCALE, photo=None):
    seed = seed if seed is not None else random.randint(1, 10000)
    generator = torch.Generator(device=device).manual_seed(seed)

    if photo is not None:
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = init_image.resize((width, height))
        with autocast(device):
            image = img2imgPipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)["images"][0]
    else:
        with autocast(device):
            image = pipe(prompt=prompt, height=height, width=width, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)["images"][0]
    return image, seed

# Handler for /gen command
async def generate_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text("Please provide a prompt after the command. Example: /gen mysterious forest")
        return

    user_prompt = ' '.join(context.args)
    predefined_prompt = "bigfoot, snow"  # Predefined prompt to prepend
    combined_prompt = f"{predefined_prompt}, {user_prompt}"
    
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    im, seed = generate_image(prompt=combined_prompt)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.effective_user.id, image_to_bytes(im), caption=f'"{combined_prompt}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)

app = ApplicationBuilder().token(TG_TOKEN).build()

app.add_handler(CommandHandler("gen", generate_command_handler))

app.run_polling()
