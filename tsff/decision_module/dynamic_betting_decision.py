import numpy as np
from scipy.stats import norm

class base_decision_module:
    def __init__(self,batch = True):
        self.batch = batch 

    def decide(self,prediction, cur_price):
        # cur_price: [time_idx,1] - if it is online, shape = [1,1] 
        # prediction: [time_idx,h] - h is prediction windows, t+1, ..., t+h
        
        buy_prob = prediction[:,2]
        sell_prob = prediction[:,0]

        buy_z = (2* buy_prob - 1) / (2* np.sqrt(buy_prob * (1- buy_prob)))
        sell_z = (2* sell_prob - 1) / (2* np.sqrt(sell_prob * (1- sell_prob)))

        buy_size = norm.cdf(buy_z)
        sell_size = norm.cdf(sell_z)

        cls_pred = np.argmax(prediction,axis = 1)

        buy_decision = np.where(cls_pred == 2, 1,0) * buy_size
        sell_decision = np.where(cls_pred == 0, 1, 0) * sell_size

        return buy_decision, sell_decision 


