import requests
import dataclasses
from typing import List
import datetime
from metrics.metric import AvgMetric
import ast

import asyncio

FLOWER_URL = 'http://localhost:5555'


@dataclasses.dataclass
class TaskRes:
    runtime: str
    result: str
    name: str
    started: int
    succeeded: int
    queued: int
    priority: str


QUEUE_SIZE = AvgMetric('Queue Size')

WORKERS_COUNT = AvgMetric('Workers Count')
PRIORITY_WORKERS_COUNT = AvgMetric('Priority Workers Count')
NOT_PRIORITY_WORKERS_COUNT = AvgMetric('Not Priority Workers Count')

RESPONSE_TIME = AvgMetric('Response Time')
PRIORITY_RESPONSE_TIME = AvgMetric('Priority Response Time')
NOT_PRIORITY_RESPONSE_TIME = AvgMetric('Not Priority Response Time')

QUEUE_TIME = AvgMetric('Queue Time')
PRIORITY_QUEUE_TIME = AvgMetric('Priority Queue Time')
NOT_PRIORITY_QUEUE_TIME = AvgMetric('Not Priority Queue Time')

DELTA_SECONDS = 1  # one second
CUR_DATETIME = datetime.datetime.now()

CUR_DATETIME.isoformat()


def _get_tasks(state, received_start=None, received_end=None):
    params = {"state": state}

    if received_end is not None:
        params["received_start"] = received_start
    if received_end is not None:
        params['received_end'] = received_end

    resp = requests.get(f'{FLOWER_URL}/api/tasks', params)

    res = resp.json()
    arr = []

    for _, value in res.items():
        arr.append(TaskRes(
            result=value['result'],
            runtime=None if value['runtime'] is None else float(value['runtime']),
            name=value['name'],
            priority='high' if value['name'] == 'tasks.priority' else 'low',
            succeeded=None if value['succeeded'] is None else float(value['succeeded']),
            started=float(value['started']),
            queued=datetime.datetime.fromisoformat(
                ast.literal_eval(value['result'])['creation_date']
            ).timestamp() if state == 'SUCCESS' else None,
        ))

    return arr


async def collect_succeeded_tasks():
    sleep_time = 60

    while True:

        res: List[TaskRes] = _get_tasks(
            state='SUCCESS',
            received_start=CUR_DATETIME.strftime('%Y-%m-%d %H:%M'),
            received_end=(CUR_DATETIME + datetime.timedelta(0, 60)).strftime('%Y-%m-%d %H:%M')
        )

        priority_tasks = []
        not_priority_tasks = []

        for i in res:
            if i.priority == 'high':
                priority_tasks.append(i)
            else:
                not_priority_tasks.append(i)

        for item in res:
            RESPONSE_TIME.collect(item.succeeded - item.queued)
            QUEUE_TIME.collect(item.started - item.queued)

        for item in priority_tasks:
            PRIORITY_RESPONSE_TIME.collect(item.succeeded - item.queued)
            PRIORITY_QUEUE_TIME.collect(item.started - item.queued)

        for item in not_priority_tasks:
            NOT_PRIORITY_RESPONSE_TIME.collect(item.succeeded - item.queued)
            NOT_PRIORITY_QUEUE_TIME.collect(item.started - item.queued)

        await asyncio.sleep(sleep_time)

        print_metrics()


async def collect_running_tasks():
    time_to_start = datetime.datetime.now()
    sleep_time = 0.3

    while True:
        res: List[TaskRes] = _get_tasks(
            state='STARTED',
        )

        high = []
        low = []

        for item in res:
            if item.priority == 'high':
                high.append(item)
            else:
                low.append(item)

        WORKERS_COUNT.collect(len(res))
        PRIORITY_WORKERS_COUNT.collect(len(high))
        NOT_PRIORITY_WORKERS_COUNT.collect(len(low))

        await asyncio.sleep(sleep_time)


async def collect_queue_size():
    sleep_time = 0.3

    while True:
        resp = requests.get(f'{FLOWER_URL}/api/queues/length')
        res = resp.json()
        # only from one queue (by default celery)
        QUEUE_SIZE.collect(int(res["active_queues"][0]["messages"]))
        await asyncio.sleep(sleep_time)


# Decorate function with metric.
def collect_metrics():
    collect_queue_size()
    collect_running_tasks()


def print_metrics():
    arr = [
        QUEUE_SIZE,
        WORKERS_COUNT,
        PRIORITY_WORKERS_COUNT,
        NOT_PRIORITY_WORKERS_COUNT,
        RESPONSE_TIME,
        PRIORITY_RESPONSE_TIME,
        NOT_PRIORITY_RESPONSE_TIME,
        QUEUE_TIME,
        PRIORITY_QUEUE_TIME,
        NOT_PRIORITY_QUEUE_TIME
    ]
    print('#######################################################')
    for item in arr:
        print(item)


async def main():
    task_running_tasks = asyncio.create_task(collect_running_tasks())
    queue_task = asyncio.create_task(collect_queue_size())
    task_success_task = asyncio.create_task(collect_succeeded_tasks())
    await asyncio.gather(task_success_task, task_running_tasks, queue_task)


if __name__ == '__main__':
    asyncio.run(main())
