class Experiment:
    def __init__(self, savefile:str):
        self.savefile = savefile
        self.data = []

    def load(self):
        file = open(self.savefile, "r")
        for i in file.readlines():
            self.data.append(
                exec(i)
            )
        file.close()
    
    def save(self, params:dict, score=None):
        if type(params) != dict:
            raise ValueError("Yikes man, should be a dictionary")
        if score:
            new_dct = {**params, "score":score}
        else:
            new_dct=params
        file = open(self.savefile, "a")
        file.write(str(new_dct)+'\n')
        file.close()