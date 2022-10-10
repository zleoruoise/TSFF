def convert_df2ts(self,data:Dict[str,pd.DataFrame]):
    result_data = []
    for key,value in data.itmes():
        cur_ts = torch.from_numpy(value.values, dtype = torch.float32)
        result_data.append(result_data)
    return result_data