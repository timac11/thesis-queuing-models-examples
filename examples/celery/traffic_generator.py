from tasks import priority, non_priority
import time
from common.mmap import Mmap
from datetime import datetime
from pyqumo.randoms import Exponential

lmb = 3 # 5 events per second
mu = lmb / 3

if __name__ == 'main':
    mmap_ = Mmap.exponential(lmb, nc=2, p=[1/3, 2/3])
    count = 0
    while True:
        sleep, cl_ = mmap_()
        n = Exponential(mu)()
        count = count + 1
        print(f'Program sleep for {sleep} secs, class: {cl_}')
        print(f'Count of generated tasks: {count}')
        time.sleep(sleep)
        if cl_ == 1:
            priority.apply_async(
                args=[n, datetime.now().isoformat()],
                priority=1
            )
        else:
            non_priority.apply_async(
                args=[n, datetime.now().isoformat()],
                priority=10
            )