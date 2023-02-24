from config import Config


class Model:
    def __init__(self,seed:int,stepsize_sim:int,stepsize_con:int,stepsize_scr:int,learning_rate:int,sample:str) -> None:
        self.seed = seed,
        self.sim = stepsize_sim,
        self.miu = stepsize_con,
        self.niu = stepsize_scr,
        self.lr = learning_rate,
        self.sample = sample

        self.stepsize_sim = stepsize_sim 
        self.stepsize_con = stepsize_con
        self.stepsize_scr = stepsize_scr
        
    
    def __str__(self) -> str:
        print("************************************************")
        print('Model description:')
        print(f'sample: {self.sample}')
        print(f'seed: {self.seed}')
        print(f'lr: {self.lr}')
        print(f'sim: {self.stepsize_sim}')
        print(f'miu: {self.stepsize_con}')
        print(f'niu: {self.stepsize_scr}')

class ModelBuilder:
    def buildModel(self)->list:
        pass
        

class BruitForceModelBuilder(ModelBuilder):
    def buildModel(self) -> list:
        models = []
        for sample in Config.samples:
            for seed in Config.seed_options:
                for sim in Config.sim_options:
                    for miu in Config.miu_options:
                        for niu in Config.niu_options:
                            for lr in Config.lr_options:
                                models.append(
                                    Model(
                                        seed = seed,
                                        stepsize_sim = sim,
                                        stepsize_con = miu,
                                        stepsize_scr = niu,
                                        learning_rate = lr,
                                        sample = sample
                                    )
                                )
        return models
