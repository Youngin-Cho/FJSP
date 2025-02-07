import torch
import simpy
import copy
import numpy as np
import pandas as pd

from collections import OrderedDict
from torch_geometric.data import HeteroData
from environment.data import DataGenerator
from environment.simulation import *
from utils.visualize import WIP_graph


class StatePDR:
    def __init__(self, num_jobs, num_machines):
        self.priority_index = np.zeros((num_jobs, num_machines))
        self.mask_pair = np.zeros((num_jobs, num_machines), dtype=bool)

    def update(self, priority_index, mask_pair):
        self.priority_index = priority_index
        self.mask_pair = mask_pair


class State:
    def __init__(self, num_jobs, num_operations, num_machines, look_ahead, device, state_encoding="DG",
                 input_dim_o=9, input_dim_j=4, input_dim_m=6, input_dim_pair=4):

        if state_encoding == "DG":
            fea_o = torch.zeros((num_operations, input_dim_o)).to(device)
            fea_m = torch.zeros((num_machines, input_dim_m)).to(device)
            edge_o_to_m = torch.from_numpy(np.array([[], []])).type(torch.long).to(device)
            edge_m_to_o = torch.from_numpy(np.array([[], []])).type(torch.long).to(device)

            self.fea_g = HeteroData()
            self.fea_g["operation"].x = fea_o
            self.fea_g["machine"].x = fea_m
            self.fea_g["operation", "operation_to_machine", "machine"].edge_index = edge_o_to_m
            self.fea_g["machine", "machine_to_operation", "operation"].edge_index = edge_m_to_o

        elif state_encoding == "BG":
            fea_j = torch.zeros((num_jobs, input_dim_j + look_ahead * input_dim_o)).to(device)
            fea_m = torch.zeros((num_machines, input_dim_m)).to(device)
            edge_j_to_m = torch.from_numpy(np.array([[], []])).type(torch.long).to(device)
            edge_m_to_j = torch.from_numpy(np.array([[], []])).type(torch.long).to(device)

            self.fea_g = HeteroData()
            self.fea_g["job"].x = fea_j
            self.fea_g["machine"].x = fea_m
            self.fea_g["job", "job_to_machine", "machine"].edge_index = edge_j_to_m
            self.fea_g["machine", "machine_to_job", "job"].edge_index = edge_m_to_j

        self.fea_pair = np.zeros((num_jobs, num_machines, input_dim_pair))
        self.mask_pair = np.zeros((num_jobs, num_machines), dtype=bool)
        self.current_o = np.zeros(num_jobs)

    def update(self, fea_g, fea_pair, mask_pair, current_o):
        self.fea_g = fea_g
        self.fea_pair = fea_pair
        self.mask_pair = mask_pair
        self.current_o = current_o


class FlexibleJobShop:
    def __init__(self, data_src, look_ahead, device, algorithm='RL', state_encoding='DG', record_events=False):
        self.data_src = data_src
        self.look_ahead = look_ahead
        self.device = device
        self.algorithm = algorithm
        self.state_encoding = state_encoding
        self.record_events = record_events

        self.df_scenario, self.df_initial, self.num_jobs, self.num_operations, self.num_machines, \
            self.job_ids, self.machine_ids, self.due_dates, self.estimated_makespan = self._initialize()

        self.input_dim_o = 9
        self.input_dim_j = 4
        self.input_dim_m = 6
        self.input_dim_pair = 4

        if self.state_encoding == "DG":
            self.meta_data = (["machine", "operation"],
                              [("operation", "predecessor", "operation"),
                               ("machine", "machine_to_operation", "operation"),
                               ("operation", "operation_to_machine", "machine")])
            self.input_dim_g = {"machine": self.input_dim_m, "operation": self.input_dim_o}
            self.num_nodes = {"machine": self.num_machines, "operation": self.num_operations}
        elif self.state_encoding == "BG":
            self.meta_data = (["machine", "job"],
                              [("machine", "machine_to_job", "job"),
                               ("job", "job_to_machine", "machine")])
            self.input_dim_g = {"machine": self.input_dim_m, "job": self.input_dim_j + look_ahead * self.input_dim_o}
            self.num_nodes = {"machine": self.num_machines, "job": self.num_jobs}

    def step(self, action):
        machine_id = action % self.num_machines
        job_id = action // self.num_machines
        done = False

        job = self.monitor.remove_queue(job_id)
        operation = job.get_current_operation()
        current_machine = job.current_machine
        next_machine = self.machine_ids[machine_id] if self.machine_ids.get(machine_id) is not None else "Buffer"

        self.completion_time_updated[job.id] = self.sim_env.now + operation.get_processing_time(machine_id)

        if current_machine is None:
            self.model["Source"].calling_event[job.name].succeed(next_machine)
        else:
            self.model[current_machine].calling_event[job.name].succeed(next_machine)

        self.actions_done.append(machine_id)

        while True:
            while True:
                if self.monitor.scheduling:
                    while self.sim_env.now in [event[0] for event in self.sim_env._queue]:
                        self.sim_env.step()
                    break
                if len(self.monitor.operations_done) == len(self.df_scenario):
                    done = True
                    break
                self.sim_env.step()

            if self.decision_time != self.sim_env.now:
                self.actions_done = []

            if self.algorithm == "RL":
                next_state = self._get_state_for_RL()
            else:
                next_state = self._get_state_for_heuristics()

            if done:
                break

            if not next_state.mask_pair.any():
                new_jobs = []
                for job in self.monitor.jobs_in_queue.values():
                    operation = job.get_current_operation()
                    if not operation.id in self.monitor.operations_in_buffer.keys():
                        new_jobs.append(job.id)

                for job_id in new_jobs:
                    job = self.monitor.remove_queue(job_id)
                    current_machine = job.current_machine
                    next_machine = "Buffer"

                    if current_machine is None:
                        self.model["Source"].calling_event[job.name].succeed(next_machine)
                    else:
                        self.model[current_machine].calling_event[job.name].succeed(next_machine)
                if self.monitor.scheduling:
                    self.monitor.scheduling = False
            else:
                break

        reward = self._calculate_reward()

        self.estimated_completion_time = copy.copy(self.estimated_completion_time_updated)
        self.completion_time = copy.copy(self.completion_time_updated)
        self.total_tardiness = self.model["Sink"].total_tardiness
        if self.decision_time != self.sim_env.now:
            self.decision_time = self.sim_env.now

        return next_state, reward, done

    def reset(self):
        self.sim_env, self.jobs, self.model, self.monitor = self._modeling()

        self.actions_done = []
        self.completion_time = np.zeros(self.num_jobs)
        self.completion_time_updated = np.zeros(self.num_jobs)
        self.estimated_completion_time = np.zeros(self.num_jobs)
        self.estimated_completion_time_updated = np.zeros(self.num_jobs)
        self.total_tardiness = 0.0
        self.decision_time = 0.0

        # cnt = 1
        while True:
            while True:
                if self.monitor.scheduling:
                    while self.sim_env.now in [event[0] for event in self.sim_env._queue]:
                        self.sim_env.step()
                    break
                self.sim_env.step()

            if self.decision_time != self.sim_env.now:
                self.actions_done = []
                self.decision_time = self.sim_env.now

            if self.algorithm == "RL":
                initial_state = self._get_state_for_RL()
            else:
                initial_state = self._get_state_for_heuristics()

            if not initial_state.mask_pair.any():
                new_jobs = []
                for job in self.monitor.jobs_in_queue.values():
                    operation = job.get_current_operation()
                    if not operation.id in self.monitor.operations_in_buffer.keys():
                        new_jobs.append(job.id)

                if len(new_jobs) != 0:
                    for job_id in new_jobs:
                        job = self.monitor.remove_queue(job_id)
                        current_machine = job.current_machine
                        next_machine = "Buffer"

                        if current_machine is None:
                            self.model["Source"].calling_event[job.name].succeed(next_machine)
                        else:
                            self.model[current_machine].calling_event[job.name].succeed(next_machine)
                else:
                    self.monitor.scheduling = False
            else:
                self.decision_time = self.sim_env.now
                break

        self.estimated_completion_time = copy.copy(self.estimated_completion_time_updated)

        return initial_state

    def _initialize(self):
        if type(self.data_src) is DataGenerator:
            df_scenario, df_initial = self.data_src.generate()
            # flag = True
            # while flag:
            #     df_scenario, df_initial = self.data_src.generate()
            #     max_wip = WIP_graph(df_scenario)
            #     if max_wip <= 20:
            #         flag = False
        else:
            df_scenario = pd.read_excel(self.data_src, sheet_name="scenario", engine='openpyxl')
            df_initial = pd.read_excel(self.data_src, sheet_name="initial", engine='openpyxl')

        df_scenario = df_scenario.sort_values(by=["Operation_Index"])

        num_jobs = len(df_scenario["Job_Name"].unique())
        num_operations = len(df_scenario)
        num_machines = len(df_scenario.columns[9:])

        job_ids = OrderedDict()
        for i, name in zip(df_scenario["Job_Index"].unique(), df_scenario["Job_Name"].unique()):
            job_ids[int(i)] = name

        machine_ids = OrderedDict()
        for i, name in enumerate(df_scenario.columns[9:]):
            machine_ids[int(i)] = name

        due_dates = np.zeros(num_jobs)
        for i, due_date in zip(df_scenario["Job_Index"].unique(), df_scenario["Due_Date"].unique()):
            due_dates[int(i)] = due_date

        estimated_makespan = 0.0
        df_scenario_group = df_scenario.groupby(by=["Job_Name", "Job_Index", "Arrival_Date", "Due_Date"])
        for idx, group in df_scenario_group:
            job_name, job_index, arrival_date, due_date = idx

            temp = arrival_date
            for i, row in group.iterrows():
                proctime = row.iloc[9:].to_numpy()
                proctime_mean = np.mean(proctime[proctime.nonzero()])
                temp += proctime_mean

            if temp > estimated_makespan:
                estimated_makespan = temp

        return (df_scenario, df_initial, num_jobs, num_operations, num_machines,
                job_ids, machine_ids, due_dates, estimated_makespan)

    def _get_mask(self):
        mask_pairs = np.zeros((self.num_jobs,self.num_machines), dtype=bool)

        for job in self.monitor.jobs_in_process.values():
            if job.id in self.monitor.jobs_in_queue.keys():
                operation = job.get_current_operation()

                for id, name in self.machine_ids.items():
                    machine = self.model[name]

                    idle, available_time = machine.check_status()

                    processing_time = int(operation.get_processing_time(id))
                    if (processing_time != 0) and (not id in self.actions_done):
                        if idle:
                            mask_pairs[job.id, id] = 1

        return mask_pairs

    def _get_state_for_RL(self):
        if self.state_encoding == "DG":
            fea_o = np.zeros((self.num_operations, self.input_dim_o))
        elif self.state_encoding == "BG":
            fea_j = np.zeros((self.num_jobs, self.input_dim_j + self.look_ahead * self.input_dim_o))
        fea_m = np.zeros((self.num_machines, self.input_dim_m))
        fea_pair = np.zeros((self.num_jobs, self.num_machines, self.input_dim_pair))
        current_o = np.zeros(self.num_jobs)

        if self.state_encoding == "DG":
            edge_pre = [[], []]
            edge_m_to_o, edge_o_to_m = [[], []], [[], []]
        elif self.state_encoding == "BG":
            edge_m_to_j, edge_j_to_m = [[], []], [[], []]

        if len(self.monitor.jobs_completed) < self.num_jobs:
            proctime_remaining = np.zeros((self.num_operations, self.num_machines))
            proctime_current = np.zeros((self.num_jobs, self.num_machines))

            proctime_remaining_mask = np.zeros(self.num_operations, dtype=bool)
            proctime_current_mask = np.zeros(self.num_jobs, dtype=bool)

            self.estimated_completion_time_updated = copy.copy(self.estimated_completion_time)

            # Operation/Job Feature
            for j, job_name in self.job_ids.items():
                if j in self.monitor.jobs_before_arrival.keys():
                    job = self.monitor.jobs_before_arrival[j]
                    current_o[job.id] = job.operations[0].id
                elif j in self.monitor.jobs_in_process.keys():
                    job = self.monitor.jobs_in_process[j]
                    current_o[job.id] = job.operations[job.step].id
                else:
                    job = self.monitor.jobs_completed[j]
                    current_o[job.id] = job.operations[-1].id

                for operation in job.operations[job.step:]:
                    proctime_remaining[operation.id, :] = operation.options
                    proctime_remaining_mask[operation.id] = True
                if job.id in self.monitor.jobs_in_queue.keys():
                    proctime_current[job.id, :] = job.operations[job.step].options
                    proctime_current_mask[job.id] = True

                job_proctime_sum = np.sum([np.mean(operation.options[operation.options.nonzero()])
                                           for operation in job.operations])

                if job.step < len(job.operations):
                    job_remaining_proctime_sum = np.sum([np.mean(operation.options[operation.options.nonzero()])
                                                         for operation in job.operations[job.step:]])
                else:
                    job_remaining_proctime_sum = 0

                for k, operation in enumerate(job.operations):
                    flag = False
                    if k < job.step:
                        earliest_finish_time = operation.start_time
                        latest_finish_time = operation.start_time
                        mean_finish_time = operation.start_time

                        machine = self.model[operation.current_machine]
                        progress = 1.0

                        flag = True
                        proctime = operation.get_processing_time(machine.id)
                    elif k == job.step:
                        if operation.id in self.monitor.operations_in_machine.keys():
                            earliest_finish_time = operation.start_time
                            latest_finish_time = operation.start_time
                            mean_finish_time = operation.start_time

                            machine = self.model[job.current_machine]
                            progress = (self.sim_env.now - operation.start_time) / operation.options[machine.id]

                            flag = True
                            proctime = operation.get_processing_time(machine.id)
                        else:
                            earliest_finish_time = max(self.sim_env.now, job.arrival_date)
                            latest_finish_time = max(self.sim_env.now, job.arrival_date)
                            mean_finish_time = max(self.sim_env.now, job.arrival_date)

                            progress = 0.0

                    eligible_options = operation.options[operation.options.nonzero()]

                    if flag:
                        earliest_finish_time = earliest_finish_time + proctime
                        latest_finish_time = latest_finish_time + proctime
                        mean_finish_time = mean_finish_time + proctime
                    else:
                        earliest_finish_time = earliest_finish_time + np.min(eligible_options)
                        latest_finish_time = latest_finish_time + np.max(eligible_options)
                        mean_finish_time = mean_finish_time + np.mean(eligible_options)

                    # Operation Feature
                    if (operation.id in self.monitor.operations_in_machine.keys()
                            or operation.id in self.monitor.operations_in_buffer.keys()):
                        f0 = [0, 1, 0]
                    elif operation.id in self.monitor.operations_done:
                        f0 = [0, 0, 1]
                    else:
                        f0 = [1, 0, 0]

                    f1 = np.min(eligible_options) / np.max(self.df_scenario.iloc[:, 9:])
                    f2 = np.max(eligible_options) / np.max(self.df_scenario.iloc[:, 9:])
                    f3 = np.mean(eligible_options) / job_proctime_sum
                    f4 = len(eligible_options) / self.num_machines
                    f5 = earliest_finish_time / self.estimated_makespan
                    f6 = latest_finish_time / self.estimated_makespan

                    if self.state_encoding == "DG":
                        # fea_o[operation.id, :] = [f1, f2, f3, f4, f5, f6]
                        fea_o[operation.id, :3] = f0
                        fea_o[operation.id, 3:] = [f1, f2, f3, f4, f5, f6]
                    elif self.state_encoding == "BG" and k < self.look_ahead:
                        first_idx =  self.input_dim_j + k * self.input_dim_o
                        last_idx = self.input_dim_j + (k + 1) * self.input_dim_o
                        fea_j[job.id, first_idx:last_idx] = [f1, f2, f3, f4, f5, f6]

                    if self.state_encoding == "DG":
                        if k >= job.step:
                            for i, proctime in enumerate(operation.options):
                                if proctime != 0:
                                    edge_o_to_m[0].append(operation.id)
                                    edge_o_to_m[1].append(i)
                                    edge_m_to_o[0].append(i)
                                    edge_m_to_o[1].append(operation.id)
                        else:
                            machine = self.model[operation.current_machine]
                            edge_o_to_m[0].append(operation.id)
                            edge_o_to_m[1].append(machine.id)
                            edge_m_to_o[0].append(machine.id)
                            edge_m_to_o[1].append(operation.id)

                        if k > 0:
                            edge_pre[0].append(operation.id - 1)
                            edge_pre[1].append(operation.id)

                    elif self.state_encoding == "BG":
                        if k == job.step:
                            for i, proctime in enumerate(operation.options):
                                if proctime != 0:
                                    edge_j_to_m[0].append(job.id)
                                    edge_j_to_m[1].append(i)
                                    edge_m_to_j[0].append(i)
                                    edge_m_to_j[1].append(job.id)

                self.estimated_completion_time_updated[job.id] = latest_finish_time

                if self.state_encoding == "BG":
                    # Job feature
                    f1 = (len(job.operations) - job.step) # / len(job.operations)
                    f2 = job_remaining_proctime_sum # / job_proctime_sum
                    # f3 = progress
                    # f4 = (self.sim_env.now - job.waiting_start) if first_operation.id in self.monitor.operations_in_buffer.keys() else 0
                    f5 = max(earliest_finish_time - job.due_date, 0)
                    f6 = max(latest_finish_time - job.due_date, 0)

                    fea_j[job.id, 0:self.input_dim_j] = [f1, f2, f5, f6]

            # if self.state_encoding == "DG":
            #     fea_o = (fea_o - fea_o.mean(axis=0, keepdims=True)) / (fea_o.std(axis=0, keepdims=True) + 1e-8)
            # elif self.state_encoding == "BG":
            #     # Normalization
            #     fea_j = (fea_j - fea_j.mean(axis=0, keepdims=True)) / (fea_j.std(axis=0, keepdims=True) + 1e-8)
            #     # fea_j[:, 3] = fea_j[:, 3] / np.max(fea_j[:, 3]) if np.max(fea_j[:, 3]) > 0.0 else 0.0
            #     # fea_j[:, 4] = fea_j[:, 4] / np.max(fea_j[:, 5]) if np.max(fea_j[:, 5]) > 0.0 else 0.0
            #     # fea_j[:, 5] = fea_j[:, 5] / np.max(fea_j[:, 5]) if np.max(fea_j[:, 5]) > 0.0 else 0.0

            # Machine Feature
            proctime_remaining = proctime_remaining[proctime_remaining_mask]
            proctime_current = proctime_current[proctime_current_mask]

            proctime_remaining_mean = np.array([np.mean(temp[temp.nonzero()]) for temp in proctime_remaining])
            proctime_current_mean = np.array([np.mean(temp[temp.nonzero()]) for temp in proctime_current])

            proctime_remaining_sum = np.sum(proctime_remaining_mean)   # 작업이 미완료된 operation의 평균 작업시간 합
            proctime_current_sum = np.sum(proctime_current_mean)       # 스케줄링 대상 operation의 평균 작업시간 합

            available_time_list = []
            proctime_compatible = np.copy(proctime_current)

            for i, machine in enumerate(self.model.values()):
                if not "Machine" in machine.name:
                    continue

                eligible_proctime_remaining = proctime_remaining[:, machine.id][proctime_remaining[:, machine.id].nonzero()]
                eligible_proctime_current = proctime_current[:, machine.id][proctime_current[:, machine.id].nonzero()]

                idle, available_time = machine.check_status()
                available_time_list.append(available_time)

                if not idle:
                    proctime_compatible[:, machine.id] = 0

                if idle:
                    f0 = [1, 0]
                else:
                    f0 = [0, 1]

                # f1 = min(eligible_proctime_current, default=0.0)
                # f2 = np.sum(eligible_proctime_remaining) / proctime_remaining_sum
                f3 = np.sum(eligible_proctime_current) / proctime_current_sum
                # f4 = len(eligible_proctime_remaining) / len(proctime_remaining)
                f5 = len(eligible_proctime_current) / len(proctime_current)
                # f6 = 1 if idle else 0
                f7 = available_time - self.sim_env.now
                f8 = (self.sim_env.now - machine.completion_time) if idle else 0

                # fea_m[machine.id, :] = [f3, f5, f7, f8]
                fea_m[machine.id, :2] = f0
                fea_m[machine.id, 2:] = [f3, f5, f7, f8]

            # Normalization
            # fea_m = (fea_m - fea_m.mean(axis=0, keepdims=True)) / (fea_m.std(axis=0, keepdims=True) + 1e-8)
            # fea_m[:, 0] = fea_m[:, 0] / np.max(fea_m[:, 0])
            if int(np.max(available_time_list) - self.sim_env.now) != 0:
                fea_m[:, 4] = fea_m[:, 4] / (np.max(available_time_list) - self.sim_env.now)
            fea_m[:, 5] = fea_m[:, 5] / np.max(fea_m[:, 5]) if np.max(fea_m[:, 5]) > 0.0 else 0.0

            # Pair Feature
            for j, job in enumerate(self.monitor.jobs_in_queue.values()):
                current_operation = job.operations[job.step]
                for i, machine in enumerate(self.model.values()):
                    if not "Machine" in machine.name:
                        continue

                    idle, available_time = machine.check_status()
                    if idle:
                        proctime = current_operation.get_processing_time(machine.id)
                        if proctime != 0:
                            estimated_completion_time_min = self.sim_env.now
                            estimated_completion_time_max = self.sim_env.now
                            for k in range(job.step, len(job.operations)):
                                if k == job.step:
                                    estimated_completion_time_min += proctime
                                    estimated_completion_time_max += proctime
                                else:
                                    options = job.operations[k].options
                                    proctime_min = np.min(options[options.nonzero()])
                                    proctime_max = np.max(options[options.nonzero()])
                                    estimated_completion_time_min += proctime_min
                                    estimated_completion_time_max += proctime_max

                            f1 = proctime / np.max(proctime_compatible)
                            f2 = proctime / np.max(proctime_compatible[:, machine.id])
                            f3 = estimated_completion_time_min / self.estimated_makespan
                            f4 = estimated_completion_time_max / self.estimated_makespan
                            # f3 = max(estimated_completion_time_min - job.due_date, 0)
                            # f4 = max(estimated_completion_time_max - job.due_date, 0)

                            fea_pair[job.id, machine.id, :] = [f1, f2, f3, f4]

        # fea_pair = (fea_pair - fea_pair.mean(axis=0, keepdims=True)) / (fea_pair.std(axis=0, keepdims=True) + 1e-8)
        # denominator = int(max(np.max(np.abs(fea_pair[:, 2])), np.max(np.abs(fea_pair[:, 3]))))
        # if denominator != 0:
        #     fea_pair[:, 2] = fea_pair[:, 2] / denominator
        #     fea_pair[:, 3] = fea_pair[:, 3] / denominator

        if self.state_encoding == "DG":
            fea_o = torch.from_numpy(fea_o).type(torch.float32).to(self.device)
            fea_m = torch.from_numpy(fea_m).type(torch.float32).to(self.device)
            edge_o_to_m = torch.from_numpy(np.array(edge_o_to_m)).type(torch.long).to(self.device)
            edge_m_to_o = torch.from_numpy(np.array(edge_m_to_o)).type(torch.long).to(self.device)

            fea_g = HeteroData()
            fea_g["operation"].x = fea_o
            fea_g["machine"].x = fea_m
            fea_g["operation", "operation_to_machine", "machine"].edge_index = edge_o_to_m
            fea_g["machine", "machine_to_operation", "operation"].edge_index = edge_m_to_o

        elif self.state_encoding == "BG":
            fea_j = torch.from_numpy(fea_j).type(torch.float32).to(self.device)
            fea_m = torch.from_numpy(fea_m).type(torch.float32).to(self.device)
            edge_j_to_m = torch.from_numpy(np.array(edge_j_to_m)).type(torch.long).to(self.device)
            edge_m_to_j = torch.from_numpy(np.array(edge_m_to_j)).type(torch.long).to(self.device)

            fea_g = HeteroData()
            fea_g["job"].x = fea_j
            fea_g["machine"].x = fea_m
            fea_g["job", "job_to_machine", "machine"].edge_index = edge_j_to_m
            fea_g["machine", "machine_to_job", "job"].edge_index = edge_m_to_j

        # fea_pair = torch.from_numpy(fea_pair).type(torch.float32).to(self.device)

        mask_pair = self._get_mask()
        # mask_pair = torch.from_numpy(mask_pair).type(torch.bool).to(self.device)

        state = State(self.num_jobs, self.num_operations, self.num_machines,
                      self.look_ahead, self.device, self.state_encoding)
        state.update(fea_g, fea_pair, mask_pair, current_o)

        return state

    def _get_state_for_heuristics(self):
        priority_index = np.zeros((self.num_jobs, self.num_machines))

        for job in self.monitor.jobs_in_queue.values():
            operation = job.get_current_operation()
            proctimes = operation.options
            if self.algorithm == "SPT":
                priority_index[job.id, proctimes != 0] = 1 / proctimes[proctimes != 0]
            elif self.algorithm == "MWKR":
                remaining_proctime = np.sum([np.mean(temp.options[temp.options != 0]) for temp in job.operations[job.step:]])
                priority_index[job.id, proctimes != 0] = remaining_proctime
            elif self.algorithm == "MOR":
                priority_index[job.id, proctimes != 0] = len(job.operations) - job.step
            elif self.algorithm == "MDD":
                priority_index[job.id, proctimes != 0] = 1 / np.maximum(job.due_date, proctimes[proctimes != 0] + self.sim_env.now)
            elif self.algorithm == "ATC":
                priority_index[job.id, proctimes != 0] = 1 / proctimes[proctimes != 0] * np.exp(- np.maximum(job.due_date - self.sim_env.now - proctimes[proctimes != 0], 0) / proctimes[proctimes != 0])
            else:
                print("invalid algorithm name")

        mask_pair = self._get_mask()

        state = StatePDR(self.num_jobs, self.num_machines)
        state.update(priority_index, mask_pair)

        return state

    def _calculate_reward(self):
        # df_log = self.monitor.get_logs()
        # df_log = df_log[df_log["Time"] > self.decision_time]
        #
        # df_group = df_log.groupby("Location")
        #
        # working_time = 0.0
        # for location_name, df_temp in df_group:
        #     if location_name in ["Buffer", "Source", "Sink"]:
        #         continue
        #
        #     df_start = df_temp[df_temp["Event"] == "Working Started"]
        #     df_finish = df_temp[df_temp["Event"] == "Working Finished"]
        #
        #     if len(df_start) < len(df_finish):
        #         start = np.concatenate([np.array([self.decision_time]), df_start["Time"].to_numpy()])
        #         finish = df_finish["Time"].to_numpy()
        #         working_time += np.sum(finish - start)
        #     elif len(df_start) > len(df_finish):
        #         start = df_start["Time"].to_numpy()
        #         finish = np.concatenate([df_finish["Time"].to_numpy(), np.array([self.sim_env.now])])
        #         working_time += np.sum(finish - start)
        #     else:
        #         if len(df_start) == 0:
        #             if not self.model[location_name].check_status()[0]:
        #                 working_time += (self.sim_env.now - self.decision_time)
        #         else:
        #             if df_start["Time"].iloc[0] < df_finish["Time"].iloc[0]:
        #                 start = df_start["Time"].to_numpy()
        #                 finish = df_finish["Time"].to_numpy()
        #                 working_time += np.sum(finish - start)
        #             else:
        #                 start = np.concatenate([np.array([self.decision_time]), df_start["Time"].to_numpy()])
        #                 finish = np.concatenate([df_finish["Time"].to_numpy(), np.array([self.sim_env.now])])
        #                 working_time += np.sum(finish - start)
        #
        # total_time = self.num_machines * (self.sim_env.now - self.decision_time)
        # idle_time = total_time - working_time
        #
        # if total_time != 0:
        #     reward = - idle_time # / total_time
        # else:
        #     reward = 0.0

        # reward = 0.0
        # if self.sim_env.now - self.decision_time > 0:
        #     for job_id, waiting_start in self.monitor.delay.items():
        #         reward += - (self.sim_env.now - waiting_start)  # / (self.sim_env.now - self.decision_time)
        #         self.monitor.delay[job_id] = self.sim_env.now

        makespan = np.max(self.completion_time)
        makespan_updated = np.max(self.completion_time_updated)
        reward = - (makespan_updated - makespan) # / self.estimated_makespan

        # tardiness = np.sum(np.maximum(self.estimated_completion_time - self.due_dates, 0))
        # tardiness_updated = np.sum(np.maximum(self.estimated_completion_time_updated - self.due_dates, 0))
        # reward = - (tardiness_updated - tardiness)

        # reward = - np.tanh(tardiness_updated - tardiness)

        # reward = np.exp(- (self.model["Sink"].total_tardiness - self.total_tardiness)) - 1

        return reward

    def _modeling(self):
        sim_env = simpy.Environment()
        monitor = Monitor(self.record_events)

        jobs = []
        df_scenario_group = self.df_scenario.groupby(by=["Job_Name", "Job_Index", "Arrival_Date", "Due_Date"])

        for idx, group in df_scenario_group:
            job_name, job_index, arrival_date, due_date = idx

            operations = []
            for i, row in group.iterrows():
                options = np.array(row.iloc[9:].to_list())
                operation = Operation(row["Operation_Name"], row["Operation_Index"],
                                      row["Start_Date"], row["Finish_Date"], options=options)
                operations.append(operation)

            initial_step = 0
            initial_machine = None
            if job_name in self.df_initial["Job_Name"].tolist():
                initial_job = self.df_initial[self.df_initial["Job_Name"] == job_name]
                initial_step = initial_job["Order"].tolist()[0]
                initial_machine = initial_job["Initial_Machine"].tolist()[0]

            job = Job(job_name, job_index, arrival_date, due_date, operations, initial_step, initial_machine)
            jobs.append(job)

        jobs = sorted(jobs, key=lambda x: x.operations[x.step].start_expected)

        model = {}
        model["Source"] = Source(sim_env, "Source", jobs, model, monitor)
        for i, name in enumerate(self.df_scenario.columns[9:]):
            machine = Machine(sim_env, name, i, model, monitor, capacity=1)
            model[name] = machine
        model["Buffer"] = Buffer(sim_env, "Buffer", model, monitor)
        model["Sink"] = Sink(sim_env, "Sink", monitor)

        return sim_env, jobs, model, monitor