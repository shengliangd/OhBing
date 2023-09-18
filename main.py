import config
import readline
from datetime import datetime
import pickle
import sys
import select
import time
import numpy as np
import openai
from typing import Tuple, List

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class LanguageModelOpenAI:
    def __init__(self, **kwargs):
        openai.api_key = kwargs['api_key']

    def generate(self, prompt):
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
                break
            except openai.error.RateLimitError:
                logger.warning(
                    'rate limit exceeded during generation, will try again in 30s')
                time.sleep(30)
        return completion.choices[0].message.content

    def encode(self, inp: str):
        while True:
            try:
                embedding = openai.Embedding.create(
                    input=inp, model="text-embedding-ada-002")['data'][0]['embedding']
                break
            except openai.error.RateLimitError:
                logger.warning(
                    'rate limit exceeded during encoding, will try again in 30s')
                time.sleep(30)
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
    def __init__(self, path: str, lm: LanguageModel):
        self.lm = lm
        # try to load memory from path
        try:
            with open(path, 'rb') as f:
                self.memory = pickle.load(f)
        except FileNotFoundError:
            self.memory = []
        logger.debug(f'loaded memory:')
        for ts, content, _ in self.memory:
            logger.debug(f'\t{ts} {content}')

    def add(self, ts: float, content: str):
        embedding = self.lm.encode(content)
        # TODO: importance
        self.memory.append((ts, content, embedding))

    def retrieve(self, ts: float, content: str, max_num: int, thresh: float):
        embedding = self.lm.encode(content)

        logger.debug(f'retrieving for {content}')

        # retrieve according to mixed metrics: recency, relevance, and importance
        scores = []
        for ts, text, emb in self.memory:
            # recency: exponential decay
            # TODO: recency
            recency = 1
            # relevance: cosine similarity
            relevance = np.dot(embedding, emb) / \
                np.linalg.norm(embedding) / np.linalg.norm(emb)
            # TODO: importance
            importance = 1

            scores.append(recency * relevance * importance)
            logger.debug(f'\t{scores[-1]:.3f} {text}')
        # sort scores with indices, filter by thresh
        scores = np.array(scores)
        indices = np.argsort(scores)[::-1]
        indices = indices[scores[indices] > thresh]
        return [self.memory[i][:2] for i in indices[:max_num]]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def _reflect(self):
        pass

    def _cleanup(self):
        pass


class ChatBot:
    def __init__(self, memory: Memory, lm: LanguageModel, config):
        self.memory = memory
        self.lm = lm
        self.config = config

    def think(self, inp: str, chat_history: List[Tuple[str, str]]) -> str:
        # retrieve memory
        logger.debug(f'retrieving memory about: {inp}')
        mem = self.memory.retrieve(
            time.time(), inp, self.config.max_retrieve_num, self.config.similarity_thresh)
        logger.debug(f'retrieved memory: \n{mem}')

        related_memory_str = ' '.join([content for _, content in mem])
        chat_history_str = ''
        for role, text in chat_history:
            if role == 'I':
                role = self.config.name
            chat_history_str += f'{role}: {text}\n'

        # construct prompt
        prompt = f"""\
It is {datetime.now().strftime('%m/%d/%Y %H:%M')}.
{self.config.description}
{self.config.name} is having a conversation with another person.

Summary of relevant context from {self.config.name}'s memory:
{related_memory_str}

Conversation history:
{chat_history_str}

How would {self.config.name} respond?
{self.config.name}: """

        # ask LM
        logger.debug(f'generating response with prompt: \n{prompt}')
        ret = self.lm.generate(prompt).lstrip(f'{self.config.name}: ')

        return ret

    def reflect(self, chat_history: List[Tuple[str, str]]):
        if len(chat_history) == 0:
            return

        chat_history_str = ''
        for role, text in chat_history:
            if role == 'I':
                role = self.config.name
            chat_history_str += f'{role}: {text}\n'

        # construct prompt to identify notable info from the chat history
        prompt = f"""\
Below is a conversation between {self.config.name} and another person:
{chat_history_str}

What important information could be summarized from the conversation? List them in concise lines:
"""

        # ask LM
        logger.debug(f'reflecting with prompt: \n{prompt}')
        ret = self.lm.generate(prompt)

        # remove leading '*', spaces, numbering, etc.
        ret = [line.strip().lstrip('*').lstrip('0123456789.- ')
               for line in ret.split('\n') if line.strip()]

        # add these info into memory
        logger.debug(f'adding to memory: \n{ret}')
        for line in ret:
            self.memory.add(time.time(), line)


def main():
    openai.api_base = config.api_server

    language_model = LanguageModel(api_key=config.openai_key)
    memory = Memory(config.memory_path, language_model)

    bot = ChatBot(memory, language_model, config)

    # save memory on exit
    import atexit
    atexit.register(memory.save, config.memory_path)

    chat_history = []

    # reflect every chat_end_thresh times, backgroudly
    import threading

    def reflect():
        while True:
            time.sleep(config.chat_end_thresh)
            bot.reflect(chat_history)
            chat_history.clear()
    threading.Thread(target=reflect, daemon=True).start()

    while True:
        inp = input('>> ').strip()
        if inp == '':
            continue
        chat_history.append(('other', inp))

        output = bot.think(inp, chat_history)
        chat_history.append(('I', output))

        print(f'{config.name}: {output}')


if __name__ == '__main__':
    main()