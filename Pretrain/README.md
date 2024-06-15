# SeisCLIP for Pretrain

This part of code is used for Pretrain. Here is a demo for using CPU, if you want to pretrain model, please put the model and data to GPU. It is easy to revise.
Because pretrained processed used imagenet-pretrained model, it will be auto-download from Internet, if you get trouble of net, you can download it through [Google Drive](https://drive.google.com/file/d/1UIeFWl2wENr83GRtdi6Tlj4MLDZ3UctN/view?usp=drive_link). And put it in this dir "../pretrained_models/hub/checkpoints/.pth"

The different spec shape can be changed in line 53 and 70:
##### Spec 50 * 120
    train_dataset = stead_loader(...,window_length=100)
    model = AUDIO_CLIP(..., spec_tdim=120, ...).to(device)

##### Spec 50 * 600
    train_dataset = stead_loader(...,window_length=20)
    model = AUDIO_CLIP(..., spec_tdim=600, ...).to(device)


In addition, to use the pretrain code, you must put STEAD dataset, for instance chunk2.csv and chunk2.hdf5 in 'SEISCLIP/Pretrain/data'.