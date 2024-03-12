from tasks import priority, non_priority
import time
from common.mmap import Mmap
from datetime import datetime
from pyqumo.randoms import Exponential

if __name__ == '__main__':
    mmap_ = Mmap.exponential(1 / .5, nc=2)
    while True:
        sleep, cl_ = mmap_()
        n = Exponential(1 / .4)()
        print(f'Program sleep for {sleep} secs, class: {cl_}')
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
