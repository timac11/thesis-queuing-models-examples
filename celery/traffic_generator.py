import numpy as np
from tasks import priority
import time

if __name__ == '__main__':
    while True:
        n = np.random.randint(1, 10)
        print(f'Program sleep for {n} secs')
        time.sleep(n)
        priority.apply_async(
            args=[np.random.randint(1, 10), time.time()],
            priority=10
        )
