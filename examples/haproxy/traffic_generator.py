import time
from common.mmap import Mmap
from datetime import datetime
from pyqumo.randoms import Exponential
import requests
import asyncio


# generation of the requests
async def send_request(queue):
    pass


async def main():
    queue = asyncio.Queue()


# if __name__ == '__main__':
#     mmap_ = Mmap.exponential(100, nc=2)
#     while True:
#         sleep, cl_ = mmap_()
#         n = Exponential(1 / .4)()
#         print(f'Program sleep for {sleep} secs, class: {cl_}')
#         time.sleep(sleep)
#
#         if cl_ == 1:
#             requests.get('http://localhost:8080')
#         else:
#             requests.get('http://localhost:8080/priority')
