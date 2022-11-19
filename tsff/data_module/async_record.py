import os 
import time
import asyncio
import math
from datetime import datetime
import ccxt.async_support as ccxt


class data_loader:

    def __init__(self,target_list,t_int = 10,**kwargs):

        self.exchange_id = 'binance'
        self.binance_key = None
        self.binance_secret = None
        self.base_path = None
        # connect to the exchange
        self.connect()
        if 'target_pairs' in kwargs:
            self.target_pairs = kwargs['target_pairs']
            self.target_table = {key:None for key in self.target_pairs}

        self.cur_limit = 10
        self.target_list = target_list 
        self.t_int = t_int

    def connect(self):
        exchange_cls = getattr(ccxt, self.exchange_id)
        exchange = exchange_cls({'apiKey': self.binance_key, 'secret': self.binance_secret})
        self.exchange = exchange


    def main(self):
        while True:
            try:
                asyncio.run(self.overall(self.t_int))
            except Exception as e:
                print(f'error_occured_{time.localtime()}') 
                print(e)
                time.sleep(10)

    async def fetch_all(self,target):
        fetch_tasks = [self.exchange.fetch_l2_order_book(symbol = cur_target, limit = self.cur_limit) for cur_target in target]
        fetched_data = await asyncio.gather(*fetch_tasks)
        return fetched_data 

    async def write(self,datum,target_date):
        # parse datum - symbol, 10 datas - qty, pr 

        symbol = datum['symbol'] 
        bid = [str(j) for i in datum['bids'] for j in i]
        ask = [str(j) for i in datum['asks'] for j in i]
        ask = ask + bid
        lined_txt = ','.join(ask)
        symbol = datum['symbol'] 

        # make it into one liner
        lined_data = [symbol] + [str(int(time.time()*1000))] + ask + bid 
        lined_txt = ', '.join(lined_data)
        # write to file
        cur_path = os.path.join(self.base_path,symbol)
        os.makedirs(cur_path, exist_ok= True)
        cur_path = os.path.join(cur_path,target_date + '.csv')
        with open(cur_path, "a+") as f:
            f.write(lined_txt)
            f.write("\n")

    async def write_all(self,data,target_date):
        write_tasks = [self.write(datum,target_date) for datum in data]
        await asyncio.gather(*write_tasks)
        return True

    async def overall(self,t_int):
        target_list = self.target_list

        # 1. time setup
        cur_time = time.time() * 1000
        target_time = math.ceil(cur_time)
        time_till_d_end = target_time % (1000 * 3600 * 24)
        #target_date = datetime.utcfromtimestamp(int(target_time/1000)).strftime('%Y%m%d')
        target_date = datetime.utcnow().strftime('%Y%m%d')
        await asyncio.sleep(target_time - cur_time)

        while True:
            #3. fetch all
            data = await self.fetch_all(target_list)
            #4. write all
            _ = await self.write_all(data,target_date)
            # 5. daily timer - ignore daily timer
            #time_till_d_end -= t_int * 1000
            # 6. current sleep time setup
            target_time += t_int * 1000
            # 5.1 date update by cond.
            if time_till_d_end < 0:
                time_till_d_end = 3600 * 24 * 1000
            #target_date = datetime.utcfromtimestamp(int(target_time/1000)).strftime('%Y%m%d')
            target_date = datetime.utcnow().strftime('%Y%m%d')
            time_diff = target_time/1000 - time.time() 
            # fault tolerance code
            #
            print(time_diff)
            await asyncio.sleep(time_diff)

if __name__ == "__main__":
    target_pairs = ['BTCUSDT','ETHUSDT',"BNBUSDT","XRPUSDT","ADAUSDT"]
    binance = data_loader(target_list = target_pairs, t_int=0.5)
    binance.main()

       
        

        
