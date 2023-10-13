from typing import Tuple, List
import numpy as np
import time
import sys
import pickle
from datetime import datetime
import readline
import threading
import re
import math
import os
import atexit
from flask import Flask, render_template, request
import utils
import yaml
import copy
import logging
import language_models
from language_models import LanguageModel

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


cfg_name = sys.argv[1]
host = sys.argv[2]

config = yaml.load(
    open(f'configs/{sys.argv[1]}.yaml', 'r'), Loader=yaml.FullLoader)
os.makedirs(f'data/bots/{cfg_name}', exist_ok=True)


def cos_sim(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)


class Memory:
    def __init__(self, path: str, lm: LanguageModel, config):
        self.config = config
        self.lm = lm
        self.path = path
        # try to load memory from path
        try:
            with open(path, 'rb') as f:
                self.memory = pickle.load(f)
        except FileNotFoundError:
            self.memory = []
        logger.debug(f'loaded memory:')
        for ts, content, _, rating in self.memory:
            logger.debug(f'\t{ts} {rating} {content}')
        self.lock = threading.Lock()

    def add(self, ts: float, content: str):
        embedding = self.lm.encode(content)

        # importance
        prompt = f"""\
On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely important (e.g., national policy, breaking news), rate the importance of the following piece of memory:
{content}
Rating (no explanation): """
        logger.debug(f'asking for rating with prompt: \n{prompt}')
        ret = self.lm.generate(prompt)
        logger.debug(f'got rating: {ret}')
        # match rating string from the generated text
        pat = r'(\d+\.?\d*)'
        try:
            rating = float(re.findall(pat, ret)[0])
        except IndexError:
            rating = 1.0
        rating /= 10.0

        # remove very similar memories
        self.new_memory = []
        for item in copy.copy(self.memory):
            if cos_sim(item[2], embedding) < self.config['similarity_thresh']:
                self.new_memory.append(item)
        self.new_memory.append((ts, content, embedding, rating))
        self.memory = self.new_memory

    def retrieve(self, ts: float, content: str, max_num: int, thresh: float):
        embedding = self.lm.encode(content)

        logger.debug(f'retrieving for {content}')

        # retrieve according to mixed metrics: recency, relevance, and importance
        scores = []
        with self.lock:
            memory = copy.copy(self.memory)
        for ts, text, emb, importance in memory:
            # recency: exponential decay
            # decay to 0.5 in 24 hours
            tau = - (60*60*24) / np.log(0.5)
            recency = np.exp(- (time.time() - ts) / tau)
            # relevance: cosine similarity
            raw_relevance = cos_sim(embedding, emb)
            relevance = raw_relevance if raw_relevance > config['relevance_thresh'] else -math.inf

            scores.append(0.10 * recency + 0.85 *
                          relevance + 0.05 * importance)
            logger.debug(
                f'\t{scores[-1]:.3f}({recency:.3f} {raw_relevance:.3f} {importance:.3f}) {text}')
        # sort scores with indices, filter by thresh
        scores = np.array(scores)
        indices = np.argsort(scores)[::-1]
        indices = indices[scores[indices] > -math.inf]
        return [memory[i][:2] for i in indices[:max_num]]

    def reflect(self):
        with self.lock:
            memory = copy.copy(self.memory)
        mem_window_str = ""
        for ts, content, _, _ in memory[-self.config['memory_reflect_window']:]:
            mem_window_str += f'{datetime.fromtimestamp(ts).strftime("%Y/%m/%d %H:%M")}: {content}\n'

        prompt = f"""\
It is {datetime.now().strftime('%Y/%m/%d %H:%M')} now.

{mem_window_str}

List at most 3 salient high-level questions we can answer from the above statements in the same language:
"""
        logger.debug(f'reflecting memory with prompt: \n{prompt}')
        ret = self.lm.generate(prompt)
        ret = [line for line in ret.split('\n') if line.strip()]
        logger.debug(f'questions: {ret}')
        related_mems = set()
        for line in ret:
            # retrieve related memories
            mem = self.retrieve(time.time(
            ), line, self.config['max_retrieve_num'], self.config['relevance_thresh'])
            related_mems.update(mem)
        related_mems_str = ""
        for ts, content in related_mems:
            related_mems_str += f'{datetime.fromtimestamp(ts).strftime("%Y/%m/%d %H:%M")}: {content}\n'

        prompt = f"""\
It is {datetime.now().strftime('%Y/%m/%d %H:%M')} now.

{related_mems_str}

List at most 3 high-level insights that you can infer from the above statements, in the same language:
"""
        logger.debug(f'generating insights with prompt: \n{prompt}')
        ret = self.lm.generate(prompt)
        ret = [line for line in ret.split('\n') if line.strip()]
        logger.debug(f'insights: {ret}')
        # add these info into memory
        for line in ret:
            self.add(time.time(), line)

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.memory, f)


class ChatBot:
    def __init__(self, path: str, lm: LanguageModel, config):
        self.path = path

        try:
            with open(f'{path}/history.pkl', 'rb') as f:
                self.chat_history = pickle.load(f)
        except FileNotFoundError:
            self.chat_history = []

        self.current_chat = []

        self.memory = Memory(
            f'{path}/memory.pkl', lm, config)
        self.lm = lm
        self.config = config

        self.lock = threading.Lock()
        self._reflect_timer = None

    def think(self, inp: str) -> str:
        with self.lock:
            self._reflect_timer and self._reflect_timer.cancel()

            self.current_chat.append(('user', inp))
            self.chat_history.append(('user', inp))

            chat_history_str = ''
            for role, text in self.current_chat:
                if role == 'me':
                    role = self.config['name']
                chat_history_str += f'{role}: {text}\n'

            # retrieve memory
            logger.debug(f'retrieving memory about: {chat_history_str}')
            mem = self.memory.retrieve(
                time.time(), inp, self.config['max_retrieve_num'], self.config['relevance_thresh'])
            logger.debug(f'retrieved memory: \n{mem}')

            related_memory_str = ""
            for ts, content in mem:
                related_memory_str += f'{datetime.fromtimestamp(ts).strftime("%Y/%m/%d %H:%M")}: {content}\n'

            # search?
            prompt = f"""\
It is {datetime.now().strftime('%Y/%m/%d %H:%M')} now.
{self.config['name']} is having a chat:
{chat_history_str}

Q: What should we search on Internet to help {self.config['name']}? Answer with at most 5 keywords in user's language, or put "null" if not applicable.
A: """

            logger.debug(f'getting search string with prompt: \n{prompt}')
            ret = self.lm.generate(prompt).strip()
            logger.debug(f'search string: {ret}')

            if ret == "null":
                search_prompt_str = ""
            else:
                try:
                    results = utils.search(ret, 3)
                    logger.debug(f'search results: {results}')
                    if len(results) == 0:
                        search_result_str = f"found no result on the Internet."
                    else:
                        search_result_str = '\n'.join(
                            [f"---\n{title}\n{link}\n{content}" for title, content, link in results])
                except Exception as e:
                    search_result_str = f"[ERROR: {str(e)[:30]}...]\n"
                search_prompt_str = f"Real-time result about {ret} from the Internet:\n{search_result_str}"

            # response
            prompt = f"""\
It is {datetime.now().strftime('%Y/%m/%d %H:%M')} now.
{self.config['description']} {self.config['name']} has long-term memory. {self.config['name']} also leverages search result from the Internet when available.\n
"""
            prompt += f"""\
Chat history:
{chat_history_str}\n
"""
            if len(related_memory_str) > 0:
                prompt += f"""\
{self.config['name']}'s memory:
{related_memory_str}\n
"""

            prompt += f"""\
{search_prompt_str}\n
"""
            prompt += f"""\
How would {self.config['name']} respond (in markdown)?
{self.config['name']}: """

            # ask LM
            logger.debug(f'generating response with prompt: \n{prompt}')
            ret = utils.remove_prefix(self.lm.generate(
                prompt), f'{self.config["name"]}: ')
            logger.debug(f'response: {ret}')

            self.current_chat.append(('me', ret))
            self.chat_history.append(('me', ret))

            # create an alarm that will fire in chat_summary_interval seconds
            self._reflect_timer = threading.Timer(
                self.config['chat_summary_interval'], self.summarize)
            self._reflect_timer.start()

            return ret

    def summarize(self):
        with self.lock:
            if len(self.current_chat) == 0:
                return

            chat_history_str = ''
            for role, text in self.current_chat:
                if role == 'me':
                    role = self.config['name']
                chat_history_str += f'{role}: {text}\n'

            # construct prompt to identify notable info from the chat history
            prompt = f"""\
{chat_history_str}

Summarize important information from the chat above into STAND-ALONE pieces in the user's language:
"""

            # ask LM
            logger.debug(f'reflecting with prompt: \n{prompt}')
            ret = self.lm.generate(prompt)
            ret = [line for line in ret.split('\n') if line.strip()]

            # add these info into memory
            logger.debug(f'adding to memory: \n{ret}')
            for line in ret:
                self.memory.add(time.time(), line)

            self.current_chat.clear()

    def reflect(self):
        with self.lock:
            self.memory.reflect()

    def save(self):
        with open(f'{self.path}/history.pkl', 'wb') as f:
            pickle.dump(self.chat_history, f)
        self.memory.save()


language_model = language_models.create('openai')
bot = ChatBot(f'data/bots/{cfg_name}', language_model, config)

# save memory on exit
atexit.register(bot.save)

app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat_history")
def get_chat_history():
    return {'chat_history': bot.chat_history}


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    output = bot.think(userText)

    return output


@app.route("/reflect")
def reflect():
    bot.reflect()
    return "OK"


@app.route("/memory")
def render_memory():
    mem = [(datetime.fromtimestamp(m[0]).strftime("%Y/%m/%d %H:%M"),
            m[1], m[2], m[3]) for m in bot.memory.memory]
    return render_template("memory.html", memory=mem)


app.run(host=host)
