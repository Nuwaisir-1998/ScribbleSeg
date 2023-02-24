from config import Config

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
                                    {
                                        'seed': seed,
                                        'stepsize_sim': sim,
                                        'stepsize_con': miu,
                                        'stepsize_scr': niu,
                                        'lr': lr,
                                        'sample': sample
                                    }
                                )
        return models
