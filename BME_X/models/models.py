
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'seg':
        assert(opt.dataset_mode == 'seg')
        from .DUNet3D_seg_recon_softmax import DenseUNet3d
        model = DenseUNet3d()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    #model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
