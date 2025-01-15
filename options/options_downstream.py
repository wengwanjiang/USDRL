root_path = "./data"
root_path = "/data/wwj_bkup/UmURL0711/data/"

class opts_pku_v1_xsub():

  def __init__(self):
    self.name = 'pkuv1Xsub'
    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "hidden_size":1024,
    "num_head":1,
    "num_layer":2,
    "num_class": 51 + 1, # background
    'alpha' : 0.5,
    'kernel_size': 1,
    'gap': 4,
  }

    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "PKUv1_xsub_train.pkl",
      'l_ratio': [1.0],
      'input_size': 64
    }
   
    self.test_feeder_args = {
      "data_path": root_path + "PKUv1_xsub_val.pkl",
      'l_ratio': [1.0],
      'input_size': 64
    }
class opts_pku_v2_xsub():

  def __init__(self):
    self.name = 'pkuv2Xsub'
    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "hidden_size":1024,
    "num_head":1,
    "num_layer":2,
    "num_class": 51,
    'alpha' : 0.5,
    'kernel_size': 1,
    'gap': 4,
  }

    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "PKUv2_xsub_train.pkl",
      'l_ratio': [0.5, 1.0],
      'input_size': 64
    }
   
    self.test_feeder_args = {
      "data_path": root_path + "PKUv2_xsub_val.pkl",
      'l_ratio': [1.0],
      'input_size': 64
    }


class  opts_ntu_60_cross_view():
  def __init__(self):
    self.name = 'ntu60Xview'
    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "hidden_size":1024,
    "num_head":1,
    "num_layer":2,
    "num_class":60,
    'alpha' : 0.5,
    'kernel_size': 1,
    'gap': 4,
   }
  
    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "/NTU-RGB-D-60-AGCN/xview/train_data_joint.npy",
      "label_path": root_path + "/NTU-RGB-D-60-AGCN/xview/train_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-60-AGCN/xview/train_num_frame.npy",
      'l_ratio': [1.0],
      'input_size': 64
    }
   
    self.test_feeder_args = {
      'data_path': root_path + "/NTU-RGB-D-60-AGCN/xview/val_data_joint.npy",
      'label_path': root_path + "/NTU-RGB-D-60-AGCN/xview/val_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-60-AGCN/xview/val_num_frame.npy",
      'l_ratio': [1.0],
      'input_size': 64
    }

class  opts_ntu_60_cross_subject():

  def __init__(self):
    self.name='ntu60Xsub'
    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "hidden_size":1024,
    "num_head":1,
    "num_layer":2,
    "num_class":60,
    'alpha': 0.5,
    'kernel_size' : 1,
    'gap': 4,  
    }

    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "/NTU-RGB-D-60-AGCN/xsub/train_data_joint.npy",
      "label_path": root_path + "/NTU-RGB-D-60-AGCN/xsub/train_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-60-AGCN/xsub/train_num_frame.npy",
      'l_ratio': [1.0],
      'input_size': 64
    }
   
    self.test_feeder_args = {
      'data_path': root_path + "/NTU-RGB-D-60-AGCN/xsub/val_data_joint.npy",
      'label_path': root_path + "/NTU-RGB-D-60-AGCN/xsub/val_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-60-AGCN/xsub/val_num_frame.npy",
      'l_ratio': [1.0],
      'input_size': 64
    }



class  opts_ntu_120_cross_subject():
  def __init__(self):
    self.name='ntu120Xsub'
    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "hidden_size":1024,
    "num_head":1,
    "num_layer":2,
    "num_class":120,
    'alpha' : 0.5,
    'kernel_size': 1,
    'gap': 4,
   }

    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "/NTU-RGB-D-120-AGCN/xsub/train_data_joint.npy",
      "label_path": root_path + "/NTU-RGB-D-120-AGCN/xsub/train_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-120-AGCN/xsub/train_num_frame.npy",
      'l_ratio': [1.0],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': root_path + "/NTU-RGB-D-120-AGCN/xsub/val_data_joint.npy",
      'label_path': root_path + "/NTU-RGB-D-120-AGCN/xsub/val_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-120-AGCN/xsub/val_num_frame.npy",
      'l_ratio': [1.0],
      'input_size': 64
    }

class  opts_ntu_120_cross_setup():

  def __init__(self):
    self.name='ntu120Xset'
    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "hidden_size":1024,
    "num_head":1,
    "num_layer":2,
    "num_class":120,
    'alpha' : 0.5,
    'kernel_size': 1,
    'gap': 4,
    }
    
    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "/NTU-RGB-D-120-AGCN/xsetup/train_data_joint.npy",
      "label_path": root_path + "/NTU-RGB-D-120-AGCN/xsetup/train_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-120-AGCN/xsetup/train_num_frame.npy",
      'l_ratio': [1.0],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': root_path + "/NTU-RGB-D-120-AGCN/xsetup/val_data_joint.npy",
      'label_path': root_path + "/NTU-RGB-D-120-AGCN/xsetup/val_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-120-AGCN/xsetup/val_num_frame.npy",
      'l_ratio': [1.0],
      'input_size': 64
    }
