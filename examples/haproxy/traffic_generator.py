from common.mmap import Mmap
import requests_async as requests
import asyncio
import datetime
from common.metrics.metric import AvgMetric, CountMetric

PACKETS_COUNT = CountMetric('Packets Count')
PRIORITY_PACKETS_COUNT = CountMetric('Priority Packets Count')
NOT_PRIORITY_PACKETS_COUNT = CountMetric('Not Priority Packets Count')

LOSS_PACKETS = CountMetric('Loss Packets')
PRIORITY_LOSS_PACKETS = CountMetric('Priority Loss Packets')
NOT_PRIORITY_LOSS_PACKETS = CountMetric('Not Priority Loss Packets')

RESPONSE_TIME = AvgMetric('Response Time')
PRIORITY_RESPONSE_TIME = AvgMetric('Priority Response Time')
NOT_PRIORITY_RESPONSE_TIME = AvgMetric('Not Priority Response Time')

QUEUE_TIME = AvgMetric('Queue Time')
PRIORITY_QUEUE_TIME = AvgMetric('Priority Queue Time')
NOT_PRIORITY_QUEUE_TIME = AvgMetric('Not Priority Queue Time')

SERVICE_TIME = AvgMetric('Service Time')
PRIORITY_SERVICE_TIME = AvgMetric('Priority Service Time')
NOT_PRIORITY_SERVICE_TIME = AvgMetric('Not Priority Service Time')

PROXY_URL = 'http://localhost:8080'


def collect_packet_statistic(res, start, is_priority=False):
    PACKETS_COUNT.collect()

    if res.status_code == 200:
        resp = res.json()
        service_time = (
            datetime.datetime.fromisoformat(resp['completed']) -
            datetime.datetime.fromisoformat(resp['started'])
        ).total_seconds()

        response_time = (datetime.datetime.fromisoformat(resp['completed']) - start).total_seconds() + 3600 * 3
        queue_time = response_time - service_time

        RESPONSE_TIME.collect(response_time)
        QUEUE_TIME.collect(queue_time)
        SERVICE_TIME.collect(service_time)

        if is_priority:
            PRIORITY_PACKETS_COUNT.collect()
            PRIORITY_RESPONSE_TIME.collect(response_time)
            PRIORITY_QUEUE_TIME.collect(queue_time)
            PRIORITY_SERVICE_TIME.collect(service_time)
        else:
            NOT_PRIORITY_PACKETS_COUNT.collect()
            NOT_PRIORITY_RESPONSE_TIME.collect(response_time)
            NOT_PRIORITY_QUEUE_TIME.collect(queue_time)
            NOT_PRIORITY_SERVICE_TIME.collect(service_time)
    else:
        LOSS_PACKETS.collect()

        if is_priority:
            PRIORITY_LOSS_PACKETS.collect()
        else:
            NOT_PRIORITY_LOSS_PACKETS.collect()


async def priority_request():
    start = datetime.datetime.now()
    res = await requests.get(f'{PROXY_URL}/priority')
    collect_packet_statistic(res, start, is_priority=True)


async def not_priority_request():
    start = datetime.datetime.now()
    res = await requests.get(f'{PROXY_URL}/not-priority')
    collect_packet_statistic(res, start, is_priority=False)


async def print_metrics():
    arr = [
        PACKETS_COUNT,
        PRIORITY_PACKETS_COUNT,
        NOT_PRIORITY_PACKETS_COUNT,
        LOSS_PACKETS,
        PRIORITY_LOSS_PACKETS,
        NOT_PRIORITY_LOSS_PACKETS,
        RESPONSE_TIME,
        PRIORITY_RESPONSE_TIME,
        NOT_PRIORITY_RESPONSE_TIME,
        QUEUE_TIME,
        PRIORITY_QUEUE_TIME,
        NOT_PRIORITY_QUEUE_TIME,
        SERVICE_TIME,
        PRIORITY_SERVICE_TIME,
        NOT_PRIORITY_SERVICE_TIME
    ]

    while True:
        await asyncio.sleep(60)
        print('#######################################################')
        for item in arr:
            print(item)


async def main():
    mmap_ = Mmap.exponential(27, nc=2)
    tasks = []

    print_metric_task = asyncio.create_task(print_metrics())

    while True:
        sleep, cl_ = mmap_()
        await asyncio.sleep(sleep)

        if cl_ == 1:
            tasks.append(asyncio.create_task(priority_request()))
        else:
            tasks.append(asyncio.create_task(not_priority_request()))


asyncio.run(main())
