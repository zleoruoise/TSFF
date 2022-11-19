import numpy as np

#class base_decision_module:
#    def __init__(self,batch = True):
#        self.batch = batch 
#
#    def decide(self,prediction, cur_price):
#        # cur_price: [time_idx,1] - if it is online, shape = [1,1] 
#        # prediction: [time_idx,h] - h is prediction windows, t+1, ..., t+h
#
#        #max_prediction = np.max(prediction, axis = 1) # cuz it keep shifts up, just use the most recent one
#        max_prediction = prediction[:,0]
#
#        assert max_prediction.shape == cur_price.shape
#
#        buy_decision = np.less(max_prediction,cur_price)
#        sell_decision = np.invert(buy_decision)
#        buy_decision = buy_decision.astype(float)
#        sell_decision = sell_decision.astype(float)
#
#
#        # buy decision is to buy underlying assets - e.g. buy ethereum
#        return buy_decision, sell_decision 



class base_decision_module:
    def __init__(self,batch = True):
        self.batch = batch 

    def decide(self,prediction, cur_price):
        # cur_price: [time_idx,1] - if it is online, shape = [1,1] 
        # prediction: [time_idx,h] - h is prediction windows, t+1, ..., t+h


        max_prediction = np.max(prediction, axis = 1) # cuz it keep shifts up, just use the most recent one
        #max_prediction = prediction[:,0]
        #cur_price = cur_price[:,0]

        assert max_prediction.shape == cur_price.shape

        buy_decision = max_prediction > cur_price * 1.010
        sell_decision = max_prediction < cur_price * 0.993
        buy_decision = buy_decision.astype(float)
        sell_decision = sell_decision.astype(float)


        # buy decision is to buy underlying assets - e.g. buy ethereum
        return buy_decision, sell_decision 


