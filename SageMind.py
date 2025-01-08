import os
import random
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ConversationHandler, filters
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load environment variables (for storing sensitive information like the Telegram token)
load_dotenv()
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

# Load DistilBERT model and tokenizer (pre-trained for sentiment classification)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Sentiment labels for classification (SST-2 dataset: 0 - negative, 1 - positive)
sentiment_labels = ['negative', 'positive']

# Response history and logs
response_history = {}
mood_log = {}
journal_log = {}

# States for ConversationHandler
MOOD, JOURNAL = range(2)

# Affirmation list
affirmations = [
    "You are stronger than you think. 🌟",
    "Your feelings are valid and important. 💖",
    "Every day is a step forward, no matter how small. 🌱",
    "Take a deep breath. You’ve got this. 🌟",
    "You are enough just as you are. 🌸",
    "Believe in yourself – you have so much potential! 💫",
    "Your journey is unique, and you’re doing amazing. ✨",
    "You are worthy of all the good things coming your way. 🌷",
    "Don’t forget to take care of yourself. You matter. 💙",
    "You’ve faced challenges before, and you can overcome this too. 💪",
]


# Function to classify sentiment of a text input
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    sentiment = torch.argmax(outputs.logits).item()
    return sentiment_labels[sentiment]

# Generate a response based on sentiment classification
def generate_response(user_input):
    # Classify sentiment of the user's input
    sentiment = classify_sentiment(user_input)

    # Depending on the sentiment, generate a response
    if sentiment == 'positive':
        positive_responses = [
            "I'm so happy you're feeling good! Keep up the great work! 😊",
            "That's fantastic! You're doing an amazing job. 🎉",
            "It's great to hear that! You're on the right track. 💪",
            "Wow! Keep riding this positive energy. You're unstoppable! ✨",
            "I'm really glad you're feeling so great! Keep shining! 🌟",
        ]
        return random.choice(positive_responses)
    else:
        negative_responses = [
            "I'm really sorry you're feeling this way. Let's talk it through. 💖",
            "It's okay to feel down sometimes. I'm here for you. 💬",
            "I'm sorry you're going through this. You’re not alone. 🤝",
            "I hear you, and I understand how tough things can get. Let's work through it together. 🌱",
            "It’s okay to not feel okay sometimes. I’m here to listen. 💙",
        ]
        return random.choice(negative_responses)

# Start command
async def start(update: Update, context):
    await update.message.reply_text(
        "Hi there! 😊 I'm your mental health chatbot.\n"
        "I can help with mood tracking, journaling, affirmations, and conversations.\n"
        "Here’s what I can do:\n"
        "/start - Start a conversation\n"
        "/help - List available commands\n"
        "/mood - Track your mood\n"
        "/journal - Write a private journal entry\n"
        "/affirmation - Receive a positive affirmation\n"
        "You can also chat with me!"
    )

# Help command
async def help_command(update: Update, context):
    await update.message.reply_text(
        "Here are the things you can do:\n"
        "/start - Start a conversation\n"
        "/help - List this message\n"
        "/mood - Track your mood\n"
        "/journal - Write a journal entry\n"
        "/affirmation - Receive a positive affirmation\n"
        "You can also chat with me about your feelings!"
    )

# Mood tracking
async def mood(update: Update, context):
    await update.message.reply_text("How are you feeling today? (e.g., Happy, Sad, Anxious)")
    return MOOD

async def log_mood(update: Update, context):
    user_id = update.message.chat_id
    mood = update.message.text
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mood_log[user_id] = {"mood": mood, "timestamp": timestamp}
    await update.message.reply_text(f"Thank you for sharing! I've logged your mood as '{mood}'.")
    return ConversationHandler.END

# Journaling
async def journal(update: Update, context):
    await update.message.reply_text("Write your thoughts below. I'll keep them private.")
    return JOURNAL

async def save_journal(update: Update, context):
    user_id = update.message.chat_id
    entry = update.message.text
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if user_id not in journal_log:
        journal_log[user_id] = []
    journal_log[user_id].append({"entry": entry, "timestamp": timestamp})
    await update.message.reply_text("Thank you for sharing. I've saved your journal entry.")
    return ConversationHandler.END

# Affirmation
async def affirmation(update: Update, context):
    await update.message.reply_text(random.choice(affirmations))

# Chat handler
async def chat(update: Update, context):
    user_message = update.message.text

    try:
        bot_reply = generate_response(user_message)
    except Exception as e:
        print(f"Error generating response: {e}")
        bot_reply = "I'm here to listen. Please share more about how you're feeling."

    await update.message.reply_text(bot_reply)

# End chat and clear session
async def end_chat(update: Update, context):
    user_id = update.message.chat_id
    if user_id in response_history:
        del response_history[user_id]
    await update.message.reply_text("Thank you for sharing. Your session data has been cleared.")

# Main function to run the bot
def main():
    if not telegram_bot_token:
        print("Error: TELEGRAM_BOT_TOKEN not found in .env file.")
        return

    # Create the application
    app = ApplicationBuilder().token(telegram_bot_token).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("affirmation", affirmation))
    app.add_handler(CommandHandler("end", end_chat))
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("mood", mood)],
        states={MOOD: [MessageHandler(filters.TEXT & ~filters.COMMAND, log_mood)]},
        fallbacks=[]
    ))
    app.add_handler(ConversationHandler(
        entry_points=[CommandHandler("journal", journal)],
        states={JOURNAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, save_journal)]},
        fallbacks=[]
    ))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    # Start the bot
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
