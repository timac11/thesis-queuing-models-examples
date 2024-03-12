from celery import Celery
import numpy as np
import time

app = Celery('tasks',
             broker='redis://localhost:6379/0',
             backend='db+sqlite:///db.db'
             )
app.control.inspect()


# app = Celery('tasks', broker='amqp://user:user@localhost:5672/')

# app.conf.broker_transport_options = {
#     'priority_steps': list(range(10)),
#     'sep': ':',
#     'queue_order_strategy': 'priority',
# }


@app.task
def fib(n):
    if n == 1 or n == 2:
        return 1

    return fib(n - 1) + fib(n - 2)


@app.task
def priority(n, creation_date):
    time.sleep(n)
    return {'sleep': n, 'creation_date': creation_date}


@app.task
def non_priority(n, creation_date):
    time.sleep(n)
    return {'sleep': n, 'creation_date': creation_date}


@app.task
def factorial(n):
    res = 1
    for i in range(1, 1 + n):
        res *= i
    return res


@app.task
def run(val: int = 10):
    sleep_time = np.random.exponential(val)
    print(f'Program sleeps {sleep_time}')
    time.sleep(sleep_time)
    return sleep_time
