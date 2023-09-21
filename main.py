import copy
import yaml
import utils
from flask import Flask, render_template, request
import atexit
import logging
import os
import math
import re
import threading
import readline
from datetime import datetime
import pickle
import sys
import select
import time
import numpy as np
import openai
from typing import Tuple, List

cfg_name = sys.argv[1]
host = sys.argv[2]

config = yaml.load(
    open(f'configs/{sys.argv[1]}.yaml', 'r'), Loader=yaml.FullLoader)
os.makedirs(f'data/bots/{cfg_name}', exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f'data/bots/{cfg_name}/log.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def cos_sim(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)


class LanguageModelOpenAI:
    def __init__(self, **kwargs):
        pass

    def generate(self, prompt):
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
                break
            except openai.error.RateLimitError:
                logger.warning(
                    'rate limit exceeded during generation, will try again in 10s')
                time.sleep(10)
            except openai.error.APIError:
                logger.warning('API error, will try again in 10s')
                time.sleep(10)
        return completion.choices[0].message.content

    def encode(self, inp: str):
        while True:
            try:
                embedding = openai.Embedding.create(
                    input=inp, model="text-embedding-ada-002")['data'][0]['embedding']
                break
            except openai.error.RateLimitError:
                logger.warning(
                    'rate limit exceeded during encoding, will try again in 10s')
                time.sleep(10)
            except openai.error.APIError:
                logger.warning('API error, will try again in 10s')
                time.sleep(10)
        return embedding


class LanguageModelDummy:
    def __init__(self, **kwargs):
        pass

    def generate(self, prompt):
        return "I don't know."

    def encode(self, inp: str):
        return np.random.rand(256)


LanguageModel = LanguageModelOpenAI


class Memory:
    def __init__(self, path: str, lm: LanguageModel, config):
        self.config = config
        self.lm = lm
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
It is {datetime.now().strftime('%Y/%m/%d %H:%M')}.

{mem_window_str}

Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?
Questions:
"""
        logger.debug(f'reflecting memory with prompt: \n{prompt}')
        ret = self.lm.generate(prompt)
        ret = [line for line in ret.split('\n') if line.strip()]
        logger.debug(f'questions: {ret}')
        related_mems = set()
        for line in ret:
            # retrieve related memories
            mem = self.retrieve(time.time(), line, self.config['max_retrieve_num'], self.config['relevance_thresh'])
            related_mems.update(mem)
        related_mems_str = ""
        for ts, content in related_mems:
            related_mems_str += f'{datetime.fromtimestamp(ts).strftime("%Y/%m/%d %H:%M")}: {content}\n'

        prompt = f"""\
It is {datetime.now().strftime('%Y/%m/%d %H:%M')}.

{related_mems_str}

What 3 high-level insights can you infer from the above statements? List them.
Insights:
"""
        logger.debug(f'generating insights with prompt: \n{prompt}')
        ret = self.lm.generate(prompt)
        ret = [line for line in ret.split('\n') if line.strip()]
        logger.debug(f'insights: {ret}')
        # add these info into memory
        for line in ret:
            self.add(time.time(), line)


    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)


class ChatBot:
    def __init__(self, memory: Memory, lm: LanguageModel, config):
        self.chat_history: List[Tuple[str, str]] = []
        self.memory = memory
        self.lm = lm
        self.config = config

        self.lock = threading.Lock()
        self._reflect_timer = None

    def think(self, inp: str) -> str:
        with self.lock:
            self._reflect_timer and self._reflect_timer.cancel()

            self.chat_history.append(('user', inp))

            chat_history_str = ''
            for role, text in self.chat_history:
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
It is {datetime.now().strftime('%Y/%m/%d %H:%M')}.
Conversation:
{chat_history_str}

If {self.config['name']} need to search on Internet for more information, provide keywords in their languange, else put "null".
keywords: """

            logger.debug(f'getting search string with prompt: \n{prompt}')
            ret = self.lm.generate(prompt)
            logger.debug(f'search string: {ret}')

            search_result = [] if "null" == ret.strip() else utils.search(ret, 3)
            logger.debug(f'search result: {search_result}')
            search_result_str = ""
            for title, content, link in search_result:
                search_result_str += f"---\n{title}\n{content}\n"

            # response
            prompt = f"""\
It is {datetime.now().strftime('%Y/%m/%d %H:%M')}.
{self.config['description']}

Conversation history:
{chat_history_str}

{self.config['name']}'s relevant memory:
{related_memory_str}

Related information abstract on the Internet:
{search_result_str}

How would {self.config['name']} respond?
{self.config['name']}: """

            # ask LM
            logger.debug(f'generating response with prompt: \n{prompt}')
            ret = utils.remove_prefix(self.lm.generate(
                prompt), f'{self.config["name"]}: ')
            logger.debug(f'response: {ret}')

            self.chat_history.append(('me', ret))

            # create an alarm that will fire in chat_summary_interval seconds
            self._reflect_timer = threading.Timer(
                self.config['chat_summary_interval'], self.summarize)
            self._reflect_timer.start()

            return ret

    def summarize(self):
        with self.lock:
            if len(self.chat_history) == 0:
                return

            chat_history_str = ''
            for role, text in self.chat_history:
                if role == 'me':
                    role = self.config['name']
                chat_history_str += f'{role}: {text}\n'

            # construct prompt to identify notable info from the chat history
            prompt = f"""\
{chat_history_str}

What facts/events/data should {self.config['name']} memorize from this conversation? List in self-contained text:
"""

            # ask LM
            logger.debug(f'reflecting with prompt: \n{prompt}')
            ret = self.lm.generate(prompt)
            ret = [line for line in ret.split('\n') if line.strip()]

            # add these info into memory
            logger.debug(f'adding to memory: \n{ret}')
            for line in ret:
                self.memory.add(time.time(), line)

            self.chat_history.clear()

    def reflect(self):
        with self.lock:
            self.memory.reflect()


openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com")
openai.api_key = os.environ["OPENAI_API_KEY"]

language_model = LanguageModel()
memory = Memory(
    f'data/bots/{cfg_name}/memory.pkl', language_model, config)
bot = ChatBot(memory, language_model, config)

# save memory on exit
atexit.register(memory.save, f'data/bots/{cfg_name}/memory.pkl')

app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


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
            m[1], m[2], m[3]) for m in memory.memory]
    return render_template("memory.html", memory=mem)


app.run(host=host)
