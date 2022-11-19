import numpy as np

class zero_one_cls_decision:
    def __init__(self,batch = True):
        self.batch = batch 

    def decide(self,prediction, cur_price):
        # cur_price: [time_idx,1] - if it is online, shape = [1,1] 
        # prediction: [time_idx,h] - h is prediction windows, t+1, ..., t+h

        # change when predictent is saved as raw
        #cls_pred = np.argmax(prediction,axis = 1)
        cls_pred = prediction

        buy_decision = np.where(cls_pred == 2, 1,0)
        sell_decision = np.where(cls_pred == 0, 1, 0)

        return buy_decision, sell_decision 


