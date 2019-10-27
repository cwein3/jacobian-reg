import os
import numpy as np 
import eval_loss_util
import matplotlib.pyplot as plt 
import json
import torch
import shutil
import ast
import models 
import pickle

def load_args(save_dir):
	args_file = os.path.join(save_dir, "args.txt")
	with open(args_file, "r") as f:
		args = f.readline()
	args = ast.literal_eval(args)
	return args

def load_model(save_dir, save_file):
	args = load_args(save_dir)
	arch = args["arch"]
	model_args = {
		"num_classes": 10 if args["dataset"] == "cifar10" else 100
	}
	model = models.__dict__[arch](**model_args)
	model_file = os.path.join(save_dir, save_file)
	model.load_state_dict(torch.load(model_file)['state_dict'])
	return model 

def load_model_sdict(model, model_file):
	model.load_state_dict(torch.load(model_file)['state_dict'])
	return model

def load_scalar_dict(save_dir):
	scalar_dict_loc = os.path.join(save_dir, "scalar_dict.pkl")
	return torch.load(scalar_dict_loc)

def write_args(args, save_dir):
	args = vars(args)
	args_file = os.path.join(save_dir, "args.txt")
	with open(args_file, "w") as args_f:
		args_f.write(str(args))

def log_scalar_file(scalar_dict, epoch, filename):
	with open(filename, "a") as f:
		f.write("Epoch %d, Data %s\n" % (epoch, json.dumps(scalar_dict)))

def save_checkpoint(
	model_state,
	data_state,
	is_best,
	save_dir):
	
	filename = os.path.join(save_dir, 'checkpoint.pth.tar')
	torch.save(model_state, filename)
	if is_best:
		shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))

	data_loc = os.path.join(save_dir, 'scalar_dict.pkl')
	torch.save(data_state, data_loc)