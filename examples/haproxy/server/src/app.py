from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import datetime
from pyqumo.randoms import Exponential

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/not-priority')
def not_priority():
    n = Exponential(1 / 3)()
    res = {'started': datetime.datetime.now().isoformat()}
    time.sleep(n)
    res['completed'] = datetime.datetime.now().isoformat()

    return res


@app.get('/priority')
def priority():
    n = Exponential(1 / 3)()
    res = {'started': datetime.datetime.now().isoformat()}
    time.sleep(n)
    res['completed'] = datetime.datetime.now().isoformat()

    return res
