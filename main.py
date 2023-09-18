import sys
import select
import time

import config
from memory import Memory


def input_timeout(timeout=0):
    if timeout == 0:
        return input()
    i, o, e = select.select([sys.stdin], [], [], timeout)
    if i:
        return sys.stdin.readline().strip()
    else:
        raise TimeoutError()


class ChatBot:
    def __init__(self, config):
        self.memory = Memory(config.memory_path)

    def think(self, inp: str) -> str:
        return 'errr'


if __name__ == '__main__':
    bot = ChatBot(config)
    chat_history = []
    while True:
        try:
            print('>> ', end='', flush=True)
            inp = input_timeout(config.chat_end_thresh)
            chat_history.append(('other', inp))

            output = bot.think(inp)
            chat_history.append(('I', output))

            print(f'{config.name}: {output}')
        except TimeoutError:
            bot.reflect()
            chat_history.clear()
