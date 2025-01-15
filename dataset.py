def get_pretraining_set(opts):
    if 'ntu' in opts.name:
        from feeder.feeder_pretraining import Feeder
        training_data = Feeder(**opts.train_feeder_args)
    elif 'pkuv2' in opts.name:
        from feeder.feeder_v2_pretrain import Feeder
        training_data = Feeder(**opts.train_feeder_args)
    return training_data


def get_finetune_training_set(opts):

    if 'ntu' in opts.name:
        from feeder.feeder_downstream import Feeder
        data = Feeder(**opts.train_feeder_args)

    elif 'pkuv2' in opts.name:

        from feeder.feeder_v2_down import Feeder
        data = Feeder(**opts.train_feeder_args)
    elif 'pkuv1' in opts.name:

        from feeder.feeder_v1_train import Feeder
        data = Feeder(**opts.train_feeder_args)

    return data

def get_finetune_validation_set(opts):

    if 'ntu' in opts.name:
        from feeder.feeder_downstream import Feeder
        data = Feeder(**opts.test_feeder_args)

    elif 'pkuv2' in opts.name:

        from feeder.feeder_v2_down import Feeder
        data = Feeder(**opts.test_feeder_args)
    elif 'pkuv1' in opts.name:

        from feeder.feeder_v1_val import Feeder
        data = Feeder(**opts.test_feeder_args)
    return data

