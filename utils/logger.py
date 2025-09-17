import copy
import os
import csv
from utils.conf import base_path
from utils.util import create_if_not_exists
useless_args = ['pub_aug','public_len','public_dataset','structure', 'model', 'csv_log', 'device_id', 'seed',
                'tensorboard','conf_jobnum','conf_timestamp','conf_host']
import pickle
from datetime import datetime


class CsvWriter:
    def __init__(self, args, private_dataset):
        self.args = args
        self.private_dataset = private_dataset
        self.model_folder_path = self._model_folder_path()
        self.para_foloder_path = self._write_args()
        print(self.para_foloder_path)

    def _model_folder_path(self):
        args = self.args
        data_path = base_path() + args.dataset
        create_if_not_exists(data_path)

        model_path = os.path.join(data_path, args.model)
        create_if_not_exists(model_path)
        return model_path

    def _write_args(self) -> str:
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.model_folder_path, current_time)
        create_if_not_exists(path)

        
        args = copy.deepcopy(self.args)
        args = vars(args)
        for cc in useless_args:
            if cc in args:
                del args[cc]

        for key, value in args.items():
            args[key] = str(value)

        args_path = os.path.join(path, 'args.csv')
        with open(args_path, 'w', newline='') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=args.keys())
            writer.writeheader()
            writer.writerow(args)

        return path

    def write_acc(self, accs_dict, mean_acc_list):
        acc_path = os.path.join(self.para_foloder_path, 'all_acc.csv')
        self._write_all_acc(acc_path, accs_dict)

        mean_acc_path = os.path.join(self.para_foloder_path, 'mean_acc.csv')
        self._write_mean_acc(mean_acc_path, mean_acc_list)

    def _write_mean_acc(self, mean_path, acc_list):
        if os.path.exists(mean_path):
            with open(mean_path, 'a') as result_file:
                result_file.write(','.join(map(str, acc_list)) + '\n')
        else:
            with open(mean_path, 'w') as result_file:
                headers = [f'epoch_{epoch}' for epoch in range(self.args.communication_epoch)]
                result_file.write(','.join(headers) + '\n')
                result_file.write(','.join(map(str, acc_list)) + '\n')

    def _write_all_acc(self, all_path, all_acc_list):
        if os.path.exists(all_path):
            with open(all_path, 'a') as result_file:
                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    result_file.write(','.join(map(str, method_result)) + '\n')
        else:
            with open(all_path, 'w') as result_file:
                headers = [f'epoch_{epoch}' for epoch in range(self.args.communication_epoch)]
                result_file.write(','.join(headers) + '\n')
                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    result_file.write(','.join(map(str, method_result)) + '\n')

    def write_loss(self, loss_dict, loss_name):
        loss_path = os.path.join(self.para_foloder_path, loss_name + '.pkl')
        with open(loss_path, 'wb+') as f:
            pickle.dump(loss_dict, f)

