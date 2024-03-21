import asyncio


async def async_sleep(n):
    await asyncio.sleep(n)
    print(f'{n} secs i slept')


async def main():
    tasks = []
    while True:
        tasks.append(asyncio.create_task(async_sleep(1)))
        await asyncio.sleep(5)
    print('end')


asyncio.run(main())
