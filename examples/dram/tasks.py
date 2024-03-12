from dramatiq.brokers.redis import RedisBroker
from dramatiq.brokers.rabbitmq import RabbitmqBroker
import dramatiq
from dramatiq.results.backends import RedisBackend
from dramatiq.results import Results
from dramatiq import Worker


broker = RabbitmqBroker(url='amqp://user:user@localhost:5672')
backend = RedisBackend(url='redis://localhost:6379')
broker.add_middleware(Results(backend=backend))
dramatiq.set_broker(broker)

broker.get_queue_message_counts()

@dramatiq.actor
def print_on_success(message, result):
    print(message)
    print(result)


@dramatiq.actor(priority=100, store_results=True)
def fib(n: int) -> int:
    if n == 1 or n == 2:
        return 1

    return fib(n - 1) + fib(n - 2)


@dramatiq.actor(priority=0, store_results=True)
def priority_fib(n: int) -> int:
    if n == 1 or n == 2:
        return 1

    return fib(n - 1) + fib(n - 2)


@dramatiq.actor
def factorial(n: int) -> int:
    res = 1
    for i in range(1, 1 + n):
        res *= i
    return res

#fib.message_with_options(args=[0], options={ "broker_priority": 255,})