from tasks import sleep
import time
from common.mmap import Mmap
from datetime import datetime
from pyqumo.randoms import Exponential

if __name__ == 'main':
    lmb = 3
    mu = lmb / 3
    mmap_ = Mmap.exponential(lmb, nc=2, p=[1/3, 2/3])
    ph_ = Exponential(mu)

    while True:
        sleep_time, cl_ = mmap_()
        n = ph_(mu)()
        time.sleep(sleep_time)

        sleep.apply_async(
            args=[n, datetime.now().isoformat()],
            priority=cl_
        )
