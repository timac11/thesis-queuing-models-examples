import requests
import dataclasses
from prometheus_client import start_http_server, Summary, Histogram, Gauge
from typing import List

import random
import time

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


# Metrics to track queue size
QUEUE = Histogram('queue_size', 'Histogram queue size')
AVG_QUEUE_SIZE = Gauge('avg_queue_size', 'Avg queue size')

# Metrics to track priority queue
# PRIORITY_QUEUE = Histogram('priority_queue_size', 'Histogram priority queue size')
# AVG_PRIORITY_QUEUE_SIZE = Gauge('avg_priority_queue_size', 'Avg priority queue size')

# Metrics to track tasks in workers
WORKERS_BUSY = Histogram('workers_busy', 'Histogram workers busy')
AVG_WORKERS_BUSY = Gauge('avg_workers_busy', 'Avg workers busy')

# Metrics to track priority tasks in workers
PRIORITY_WORKERS_BUSY = Histogram('priority_workers_busy', 'Histogram priority workers busy')
PRIORITY_AVG_WORKERS_BUSY = Gauge('priority_avg_workers_busy', 'Avg priority workers busy')

# Times metrics
RESPONSE_TIME = Gauge('response_time', 'Response time for system')
PRIORITY_RESPONSE_TIME = Gauge('priority_response_time', 'Priority response time for system')

QUEUE_TIME = Gauge('queue_time', 'Queue time for system')
PRIORITY_QUEUE_TIME = Gauge('priority_queue_time', 'Priority queue time for system')

SERVICE_TIME = Gauge('service_time', 'Service time for system')
PRIORITY_SERVICE_TIME = Gauge('priority_service_time', 'Priority service time for system')


# start options for metrics collection

START_EXECUTION_TIME = time.time()  # start time to collect metrics
DELTA = 1  # one second


def _get_tasks(state):
    resp = requests.get(f'{FLOWER_URL}/api/tasks?state={state}')

    print(f'{FLOWER_URL}/api/tasks/?state={state}')
    res = resp.json()

    arr = []

    for _, value in res.items():
        print(value)
        print(value is None)
        arr.append(TaskRes(
            result=value['result'],
            runtime=None if value['runtime'] is None else int(value['runtime']),
            name=value['name'],
            priority='high' if value['name'] == 'tasks.priority' else 'low',
            succeeded=None if value['succeeded'] is None else int(value['succeeded']),
            started=int(value['started']),
            queued=int(value['started']),
        ))

    return arr


def _get_succeeded_tasks():
    res: List[TaskRes] = _get_tasks(state='SUCCESS')

    # TODO here add: response time metric, queue time metric, service time metric


def _get_running_tasks():
    res: List[TaskRes] = _get_tasks(state='STARTED')

    high = []
    low = []

    for item in res:
        if item.priority == 'high':
            high.append(item)
        else:
            low.append(item)

    PRIORITY_WORKERS_BUSY.observe(len(high))
    WORKERS_BUSY.observe(len(res))


def get_queue_size():
    pass
    # TODO add queue size to histogram


# Decorate function with metric.
def get_metrics():
    pass


if __name__ == '__main__':
    # Start up the server to expose the metrics.
    start_http_server(8000)
    # Generate some requests.
    while True:
        time.sleep(1)
        _get_running_tasks()