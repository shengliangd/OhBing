# OhBing

A NewBing-like chat bot with long-term memory.

## How it works?

It's based on OpenAI's text generation and embedding APIs.
When the user does not send any message for a while, a conversation will be summarized into pieces, and stored with text embedding.

When replying to the user, the bot first retrieves relevant memories using cosine similarity along with some other metrics, 
and searches the Internet with appropriate keywords if necessary.
The retrieved memories and search results are then added to the prompt for reply generation.

## Usage

1. Clone this repo;
2. Install required Python packages: `pip install -r requirements.txt`;
3. Install firefox-esr: `sudo apt install firefox-esr`, and install [geckodriver](https://github.com/mozilla/geckodriver) if you see compatibility warnings later;
4. Configure environment variables:
    - `OPENAI_API_BASE`: OpenAI API base URL, default to `https://api.openai.com/v1`;
    - `OPENAI_API_KEY`: OpenAI API key.
4. Run `python3 main.py <config_name> 127.0.0.1` to start the web server;
5. Open `http://127.0.0.1:5000` in your browser and start chatting.

Tips:
1. Access `http://127.0.0.1:5000/memory` to see the memory of the bot;
2. Access `http://127.0.0.1:5000/reflect` to trigger a reflection (it's slow, wait patiently);
3. Check `data/bots/<config_name>/log.log` for logs.
