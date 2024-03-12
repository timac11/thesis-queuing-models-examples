# from examples.common.mmap import Mmap
#
# mmap = Mmap.exponential(100, nc=2)
#
# print(mmap())
# print(mmap())
# print(mmap())

import json
import ast
import datetime

task = {
    'uuid': '5dbfb36f-78ee-4417-a61e-e15b729078a7',
    'name': 'tasks.priority',
    'state': 'SUCCESS',
    'received': 1710253686.254299,
    'sent': None,
    'started': 1710253686.637316,
    'rejected': None,
    'succeeded': 1710253686.799571,
    'failed': None,
    'retried': None,
    'revoked': None,
    'args': "[0.1571477221331541, '2024-03-12T17:28:06.248831']",
    'kwargs': '{}',
    'eta': None,
    'expires': None,
    'retries': 0,
    'result': "{'sleep': 0.1571477221331541, 'creation_date': '2024-03-12T17:28:06.248831'}",
    'exception': None,
    'timestamp': 1710253686.799571,
    'runtime': 0.16166774000157602,
    'traceback': None, 'exchange': None, 'routing_key': None, 'clock': 26728, 'client': None, 'root': '5dbfb36f-78ee-4417-a61e-e15b729078a7',
    'root_id': '5dbfb36f-78ee-4417-a61e-e15b729078a7', 'parent': None, 'parent_id': None, 'children': [], 'worker': 'celery@MacBook-Pro'
}

queued = datetime.datetime.fromisoformat(
    ast.literal_eval(task['result'])['creation_date']
).timestamp()

print(queued)


