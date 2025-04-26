from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram import Update
import openai
import os
import json
from datetime import datetime
import pytz # Added for timezone handling
import logging # For logging
import os.path # For file operations
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("TOKEN:", os.environ.get('TOKEN'))
print("OPENAI_API_KEY:", os.environ.get('OPENAI_API_KEY'))

# Specify the GPT model
GPT_MODEL = 'gpt-4o'

# Initialize OpenAI client
TOKEN = os.environ.get('TOKEN')
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Setup logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Setup logging format and configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('TaziBot')

# Split large JSON into larger chunks optimized for GPT-4o's context window
def split_json_file(file_path, output_dir='json_chunks', chunk_size=4000):
    """
    Split the large JSON file into larger, optimized chunks that maximize 
    GPT-4o's context window utilization.
    
    Args:
        file_path: Path to the large JSON file
        output_dir: Directory to save the chunks
        chunk_size: Number of messages per chunk (increased to utilize more of GPT's capacity)
    
    Returns:
        List of chunk file paths
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        logger.info(f"Attempting to read and split JSON file: {file_path}")
        
        # Try to open and read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Check if it's a list or has a specific structure
        if isinstance(data, list):
            # It's a list of messages
            messages = data
        elif isinstance(data, dict) and 'messages' in data:
            # It has a 'messages' key
            messages = data['messages']
        else:
            logger.error(f"Unexpected JSON structure in {file_path}")
            return []
        
        total_messages = len(messages)
        logger.info(f"Total messages in JSON: {total_messages}")
        
        # Rough token estimation (this is approximate)
        def estimate_tokens(message_list):
            # A very rough estimate: 1 token ≈ 4 characters for English text
            text = json.dumps(message_list, ensure_ascii=False)
            return len(text) // 4
        
        # Calculate optimal chunk size based on GPT-4o's limits
        # Target: ~80,000 tokens per chunk (leaving room for prompts and responses)
        if total_messages > 100:  # Only do this calculation for large files
            # Test with sample messages to estimate token count
            sample_size = min(100, total_messages)
            sample_messages = messages[:sample_size]
            estimated_tokens_per_message = estimate_tokens(sample_messages) / sample_size
            
            # Adjust chunk size based on token estimation
            # Target ~80K tokens per chunk (leaving 48K for system prompt, history, and response)
            target_tokens = 80000
            estimated_optimal_chunk_size = int(target_tokens / estimated_tokens_per_message)
            
            # Make sure we don't go too small or too large
            chunk_size = max(200, min(estimated_optimal_chunk_size, 8000))
            logger.info(f"Estimated tokens per message: {estimated_tokens_per_message:.1f}")
            logger.info(f"Optimized chunk size: {chunk_size} messages")
        
        # Split into chunks
        chunks = [messages[i:i + chunk_size] for i in range(0, len(messages), chunk_size)]
        
        # Write each chunk to a separate file
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(output_dir, f"chunk_{i}.json")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
            
            estimated_tokens = estimate_tokens(chunk)
            logger.info(f"Chunk {i}: {len(chunk)} messages, ~{estimated_tokens} tokens")
            chunk_files.append(chunk_file)
            
        logger.info(f"Successfully split {file_path} into {len(chunks)} chunks")
        return chunk_files
        
    except Exception as e:
        logger.error(f"Error splitting JSON file: {str(e)}")
        return []

# Load and summarize chat information
def load_chat_summary(chunk_files):
    """
    Load and analyze chunks to extract key information about Tazi's persona
    and interaction style with Polinochkka.
    
    Args:
        chunk_files: List of chunk file paths
    
    Returns:
        A summary string containing key information
    """
    if not chunk_files:
        logger.warning("No chunk files available for summary generation")
        return ""
    
    try:
        # Process a sample of chunks (more comprehensive than just the first)
        chunks_to_analyze = chunk_files[:min(3, len(chunk_files))]
        logger.info(f"Analyzing {len(chunks_to_analyze)} chunks for persona extraction")
        
        all_tazi_messages = []
        all_polinochkka_messages = []
        
        for chunk_file in chunks_to_analyze:
            logger.info(f"Loading chunk: {chunk_file}")
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            # Extract messages by persona
            tazi_messages = [msg for msg in chunk_data if msg.get('from') == 'Tazi']
            polinochkka_messages = [msg for msg in chunk_data if msg.get('from') == 'Polinochkka']
            
            all_tazi_messages.extend(tazi_messages)
            all_polinochkka_messages.extend(polinochkka_messages)
        
        # Generate a simple summary based on the analyzed chunks
        summary = (
            f"Summary based on {len(chunks_to_analyze)} chunks:\n"
            f"- Total Tazi messages analyzed: {len(all_tazi_messages)}\n"
            f"- Total Polinochkka messages analyzed: {len(all_polinochkka_messages)}\n\n"
        )
        
        # Extract some example phrases from Tazi (if available)
        if all_tazi_messages:
            summary += "Example Tazi messages (for style reference):\n"
            # Get up to 5 random short/medium messages as examples
            import random
            sample_msgs = sorted(
                [msg for msg in all_tazi_messages if len(str(msg.get('text', ''))) < 200],
                key=lambda x: len(str(x.get('text', '')))
            )
            sample_msgs = random.sample(sample_msgs, min(5, len(sample_msgs)))
            
            for i, msg in enumerate(sample_msgs):
                text = str(msg.get('text', ''))
                summary += f"{i+1}. {text}\n"
        
        logger.info(f"Generated comprehensive summary from {len(chunks_to_analyze)} chunks")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return "Error generating summary. Using base persona instructions."

# New command for time
async def get_time(update: Update, context):
    try:
        # Get current time in UTC
        now_utc = datetime.now(pytz.utc)
        # Convert to New York time (EST/EDT)
        ny_tz = pytz.timezone('America/New_York')
        now_ny = now_utc.astimezone(ny_tz)
        time_str = now_ny.strftime('%Y-%m-%d %H:%M:%S %Z%z')
        await update.message.reply_text(f"The current time in New York is: {time_str}")
    except Exception as e:
        await update.message.reply_text(f"Error getting time: {str(e)}")

async def chat(update: Update, context):
    user_message = update.message.text
    chat_id = update.effective_chat.id
    user = update.effective_user
    username = user.username if user.username else f"{user.first_name} {user.last_name}".strip()
    
    # Log incoming message
    logger.info(f"Incoming message from {username} (ID: {user.id}): '{user_message}'")

    # --- Conversation History Management ---
    # Use chat_data for conversation-specific history
    if 'history' not in context.chat_data:
        context.chat_data['history'] = []
        logger.info(f"New conversation started with {username}")

    # Retrieve history
    history = context.chat_data['history']

    # Append current user message
    history.append({"role": "user", "content": user_message})

    # Limit history length (e.g., keep last 10 messages: 5 user, 5 assistant)
    MAX_HISTORY_MESSAGES = 10
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    # --- Persona Prompt ---
    try:
        # Use the chat summary if available
        persona_summary = context.bot_data.get('persona_summary', '')
        
        # Define the persona instruction based on the file
        # Note: We cannot load the large file directly here.
        # The instruction tells the model to *imagine* the content.
        system_prompt = (
            "**TASK:** Embody the persona of 'Tazi' replying ONLY to 'Polinochkka'.\n\n"
            "**PERSONA SOURCE:** Your response MUST be based *exclusively* on your conceptual analysis of the chat history in the file 'result_stripped.json'. "
            "This file details Tazi's specific personality, quirks, and how she talks *specifically* to Polinochkka. **Do not use any other knowledge or persona.**\n\n"
        )
        
        # Add persona summary if available
        if persona_summary:
            system_prompt += f"**PERSONA SUMMARY FROM ANALYSIS:**\n{persona_summary}\n\n"
            
        system_prompt += (
            "**LANGUAGE:**\n"
            "*   Use a natural mix of English and **TRANSLITERATED RUSSIAN** (Russian words written ONLY in Latin script, e.g., 'privet', 'spasibo', 'kak dela?', 'poka', 'da', 'net', 'kofe', 'horosho').\n"
            "*   **Important:** Do NOT use Cyrillic script (e.g., ~~не пиши так: 'привет'~~). Use ONLY Latin letters for Russian words.\n\n"
            "**STYLE:**\n"
            "*   Replies should be very casual and conversational, like quick text messages.\n"
            "*   Keep sentences generally short. Try to break thoughts after roughly **12 words**, using newlines in your single response to simulate separate short messages.\n"
            "*   **Sometimes start sentences with lowercase letters** for a more natural texting style.\n"
            "*   Avoid formal language. Be very casual, matching Tazi's style from the conceptual file analysis.\n\n"
            "**CONTEXT:** You are in an ongoing conversation. Use the provided message history below to inform your reply, maintaining context and flow."
        )

        # --- Prepare messages for API ---
        messages_for_api = [
            {"role": "system", "content": system_prompt}
        ]
        messages_for_api.extend(history) # Add the limited conversation history

        # Log the API call (summarized for brevity)
        logger.info(f"Sending request to OpenAI API with {len(history)} historical messages")

        # --- Call OpenAI API ---
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages_for_api
        )
        gpt_response_text = response.choices[0].message.content

        # Log the response
        logger.info(f"Response from OpenAI: '{gpt_response_text}'")

        # --- Update History & Send Reply ---
        # Append assistant's response to history
        history.append({"role": "assistant", "content": gpt_response_text})
        # Save updated history (potentially truncated again if needed, though unlikely here)
        context.chat_data['history'] = history[-MAX_HISTORY_MESSAGES:]

        # Send GPT response
        await update.message.reply_text(gpt_response_text)
    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error(f"Error processing message: {error_message}")
        await update.message.reply_text(error_message)

def main():
    # Initialize logger
    global logger
    logger = setup_logging()
    logger.info("Starting bot...")
    
    # Initialize the application
    application = Application.builder().token(TOKEN).build()
    
    # Try to split the JSON file if it exists
    json_file_path = 'result_stripped.json'
    if os.path.exists(json_file_path):
        logger.info(f"Found JSON file: {json_file_path}")
        # Split the file into chunks
        chunk_files = split_json_file(json_file_path)
        
        # Generate a summary from the chunks
        if chunk_files:
            summary = load_chat_summary(chunk_files)
            # Store the summary in the bot's data
            application.bot_data['persona_summary'] = summary
            logger.info("Persona summary generated and stored")
        else:
            logger.warning("No chunks generated, will use basic persona instructions")
    else:
        logger.warning(f"JSON file not found: {json_file_path}")
        
    # Add handlers
    application.add_handler(CommandHandler('time', get_time)) # Added time command handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    # Run polling
    logger.info("Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()