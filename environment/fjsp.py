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
    def __init__(self, num_jobs, num_machines, look_ahead, device,
                 input_dim_o=4, input_dim_j=6, input_dim_m=8, input_dim_pair=2):

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

    def update(self, fea_g, fea_pair, mask_pair):
        self.fea_g = fea_g
        self.fea_pair = fea_pair
        self.mask_pair = mask_pair


class FlexibleJobShop:
    def __init__(self, data_src, look_ahead, device, algorithm='RL', record_events=False):
        self.data_src = data_src
        self.look_ahead = look_ahead
        self.device = device
        self.algorithm = algorithm
        self.record_events = record_events

        self.data, self.num_jobs, self.num_operations, self.num_machines, \
            self.job_ids, self.machine_ids, self.due_dates = self._initialize()

        self.input_dim_o = 4
        self.input_dim_j = 6
        self.input_dim_m = 8
        self.input_dim_pair = 2

        self.meta_data = (["machine", "job"],
                          [("machine", "machine_to_job", "job"), ("job", "job_to_machine", "machine")])
        self.input_dim_g = {"machine": self.input_dim_m, "job": self.input_dim_j + look_ahead * self.input_dim_o}
        self.num_nodes = {"machine": self.num_machines, "job": self.num_jobs}

    def step(self, action):
        machine_id = action % self.num_machines
        job_id = action // self.num_machines
        done = False

        job = self.monitor.remove_queue(job_id)
        current_machine = job.current_machine
        next_machine = self.machine_ids[machine_id] if self.machine_ids.get(machine_id) is not None else "Buffer"

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
                if len(self.monitor.operations_done) == len(self.data):
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

        self.completion_time = copy.copy(self.completion_time_updated)
        if self.decision_time != self.sim_env.now:
            self.decision_time = self.sim_env.now

        return next_state, reward, done

    def reset(self):
        self.sim_env, self.jobs, self.model, self.monitor = self._modeling()

        self.actions_done = []
        self.completion_time = np.zeros(self.num_jobs)
        self.completion_time_updated = np.zeros(self.num_jobs)
        self.decision_time = 0.0

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

                for job_id in new_jobs:
                    job = self.monitor.remove_queue(job_id)
                    current_machine = job.current_machine
                    next_machine = "Buffer"

                    if current_machine is None:
                        self.model["Source"].calling_event[job.name].succeed(next_machine)
                    else:
                        self.model[current_machine].calling_event[job.name].succeed(next_machine)
            else:
                self.decision_time = self.sim_env.now
                break

        self.completion_time = copy.copy(self.completion_time_updated)

        return initial_state

    def _initialize(self):
        if type(self.data_src) is DataGenerator:
            flag = True
            while flag:
                data = self.data_src.generate()
                max_wip = WIP_graph(data)
                if len(data.columns[6:]) * 0.8 <= max_wip <= len(data.columns[6:]) * 1.2:
                    flag = False
        elif type(self.data_src) is pd.DataFrame:
            data = self.data_src
        else:
            data = pd.read_excel(self.data_src, sheet_name="scenario", engine='openpyxl')

        data = data.sort_values(by=["Operation_Index"])

        num_jobs = len(data["Job_Name"].unique())
        num_operations = len(data)
        num_machines = len(data.columns[6:])

        job_ids = OrderedDict()
        for i, name in zip(data["Job_Index"].unique(), data["Job_Name"].unique()):
            job_ids[int(i)] = name

        machine_ids = OrderedDict()
        for i, name in enumerate(data.columns[6:]):
            machine_ids[int(i)] = name

        due_dates = np.zeros(num_jobs)
        for i, due_date in zip(data["Job_Index"].unique(), data["Due_Date"].unique()):
            due_dates[int(i)] = due_date

        return data, num_jobs, num_operations, num_machines, job_ids, machine_ids, due_dates

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
        fea_j = np.zeros((self.num_jobs, self.input_dim_j + self.look_ahead * self.input_dim_o))
        fea_m = np.zeros((self.num_machines, self.input_dim_m))
        fea_pair = np.zeros((self.num_jobs, self.num_machines, self.input_dim_pair))

        edge_m_to_j, edge_j_to_m = [[], []], [[], []]

        if len(self.monitor.jobs_completed) < self.num_jobs:
            proctime_remaining = []
            proctime_current = []

            self.completion_time_updated = copy.copy(self.completion_time)

            # Operation/Job Feature
            for j, job_name in self.job_ids.items():
                if j in self.monitor.jobs_before_arrival.keys():
                    job = self.monitor.jobs_before_arrival[j]
                elif j in self.monitor.jobs_in_process.keys():
                    job = self.monitor.jobs_in_process[j]
                else:
                    continue

                for operation in job.operations[job.step:]:
                    proctime_remaining.append(operation.options)
                if job.id in self.monitor.jobs_in_queue.keys():
                    proctime_current.append(job.operations[job.step].options)

                job_proctime_sum = np.sum([np.mean(operation.options[operation.options.nonzero()])
                                           for operation in job.operations])
                job_remaining_proctime_sum = np.sum([np.mean(operation.options[operation.options.nonzero()])
                                                     for operation in job.operations[job.step:]])

                first_operation = job.operations[job.step]
                first_options = first_operation.options
                if first_operation.id in self.monitor.operations_in_machine.keys():
                    machine_id = self.model[job.current_machine].id
                    earliest_finish_time = first_operation.start_time
                    latest_finish_time = first_operation.start_time
                    mean_finish_time = first_operation.start_time
                    progress = (self.sim_env.now - first_operation.start_time) / first_options[machine_id]
                else:
                    earliest_finish_time = max(self.sim_env.now, job.arrival_date)
                    latest_finish_time = max(self.sim_env.now, job.arrival_date)
                    mean_finish_time = max(self.sim_env.now, job.arrival_date)
                    progress = 0

                # Edge
                for i, proctime in enumerate(first_operation.options):
                    if proctime != 0:
                        edge_j_to_m[0].append(j)
                        edge_j_to_m[1].append(i)
                        edge_m_to_j[0].append(i)
                        edge_m_to_j[1].append(j)

                for k, operation in enumerate(job.operations[job.step:]):
                    flag = False
                    if (k == 0) and (operation.id in self.monitor.operations_in_machine.keys()):
                        flag = True
                        proctime = operation.get_processing_time(self.model[job.current_machine].id)

                    eligible_options = operation.options[operation.options.nonzero()]

                    if k < self.look_ahead:
                        # Operation Feature
                        f1 = np.min(eligible_options) / np.max(self.data.iloc[:, 6:])
                        f2 = np.max(eligible_options) / np.max(self.data.iloc[:, 6:])
                        f3 = np.mean(eligible_options) / job_proctime_sum
                        f4 = len(eligible_options) / self.num_machines

                        first_idx =  self.input_dim_j + k * self.input_dim_o
                        last_idx = self.input_dim_j + (k + 1) * self.input_dim_o
                        fea_j[job.id, first_idx:last_idx] = [f1, f2, f3, f4]

                    if flag:
                        earliest_finish_time = earliest_finish_time + proctime
                        latest_finish_time = latest_finish_time + proctime
                        mean_finish_time = mean_finish_time + proctime
                    else:
                        earliest_finish_time = earliest_finish_time + np.min(eligible_options)
                        latest_finish_time = latest_finish_time + np.max(eligible_options)
                        mean_finish_time = mean_finish_time + np.mean(eligible_options)

                self.completion_time_updated[j] = mean_finish_time

                # Job feature
                f1 = (len(job.operations) - job.step) / len(job.operations)
                f2 = job_remaining_proctime_sum / job_proctime_sum
                f3 = progress
                f4 = (self.sim_env.now - job.waiting_start) if first_operation.id in self.monitor.operations_in_buffer.keys() else 0
                f5 = max(earliest_finish_time - job.due_date, 0)
                f6 = max(latest_finish_time - job.due_date, 0)

                fea_j[job.id, 0:self.input_dim_j] = [f1, f2, f3, f4, f5, f6]

            # Normalization
            fea_j[:, 3] = fea_j[:, 3] / np.max(fea_j[:, 3]) if np.max(fea_j[:, 3]) > 0.0 else 0.0
            fea_j[:, 4] = fea_j[:, 4] / np.max(fea_j[:, 5]) if np.max(fea_j[:, 5]) > 0.0 else 0.0
            fea_j[:, 5] = fea_j[:, 5] / np.max(fea_j[:, 5]) if np.max(fea_j[:, 5]) > 0.0 else 0.0

            # Machine Feature
            proctime_remaining = np.array(proctime_remaining)
            proctime_current = np.array(proctime_current)

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

                f1 = min(eligible_proctime_current, default=0.0)
                f2 = np.sum(eligible_proctime_remaining) / proctime_remaining_sum
                f3 = np.sum(eligible_proctime_current) / proctime_current_sum
                f4 = len(eligible_proctime_remaining) / len(proctime_remaining)
                f5 = len(eligible_proctime_current) / len(proctime_current)
                f6 = 1 if idle else 0
                f7 = available_time - self.sim_env.now
                f8 = (self.sim_env.now - machine.completion_time) if idle else 0

                fea_m[machine.id, :] = [f1, f2, f3, f4, f5, f6, f7, f8]

            # Normalization
            fea_m[:, 0] = fea_m[:, 0] / np.max(fea_m[:, 0])
            if int(np.max(available_time_list) - self.sim_env.now) != 0:
                fea_m[:, 6] = fea_m[:, 6] / (np.max(available_time_list) - self.sim_env.now)
            fea_m[:, 7] = fea_m[:, 7] / np.max(fea_m[:, 7]) if np.max(fea_m[:, 7]) > 0.0 else 0.0

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
                            f1 = proctime / np.max(proctime_compatible)
                            f2 = proctime / np.max(proctime_compatible[:, machine.id])

                            fea_pair[job.id, machine.id, :] = [f1, f2]

        fea_j = torch.from_numpy(fea_j).type(torch.float32).to(self.device)
        fea_m = torch.from_numpy(fea_m).type(torch.float32).to(self.device)
        # fea_pair = torch.from_numpy(fea_pair).type(torch.float32).to(self.device)

        edge_j_to_m = torch.from_numpy(np.array(edge_j_to_m)).type(torch.long).to(self.device)
        edge_m_to_j = torch.from_numpy(np.array(edge_m_to_j)).type(torch.long).to(self.device)

        fea_g = HeteroData()
        fea_g["job"].x = fea_j
        fea_g["machine"].x = fea_m
        fea_g["job", "job_to_machine", "machine"].edge_index = edge_j_to_m
        fea_g["machine", "machine_to_job", "job"].edge_index = edge_m_to_j

        mask_pair = self._get_mask()
        # mask_pair = torch.from_numpy(mask_pair).type(torch.bool).to(self.device)

        state = State(self.num_jobs, self.num_machines, self.look_ahead, self.device)
        state.update(fea_g, fea_pair, mask_pair)

        return state

    def _get_state_for_heuristics(self):
        priority_index = np.zeros((self.num_jobs, self.num_machines))

        for job in self.monitor.jobs_in_queue.values():
            operation = job.get_current_operation()
            proctimes = operation.options
            if self.algorithm == "SPT":
                priority_index[job.id, proctimes != 0] = 1 / proctimes[proctimes != 0]
            elif self.algorithm == "MDD":
                priority_index[job.id, proctimes != 0] = 1 / np.maximum(job.due_date, proctimes[proctimes != 0] + self.sim_env.now)
            elif self.algorithm == "ATC":
                priority_index[job.id, proctimes != 0] = 1 / proctimes[proctimes != 0] * np.exp(- np.maximum(job.due_date - self.sim_env.now - proctimes[proctimes != 0], 0) / proctimes[proctimes != 0])
            elif self.algorithm == "COVERT":
                pass
            else:
                print("invalid algorithm name")

        mask_pair = self._get_mask()

        state = StatePDR(self.num_jobs, self.num_machines)
        state.update(priority_index, mask_pair)

        return state

    def _calculate_reward(self):
        tardiness = np.sum(np.maximum(self.completion_time - self.due_dates, 0))
        tardiness_updated = np.sum(np.maximum(self.completion_time_updated - self.due_dates, 0))
        reward = - (tardiness_updated - tardiness)
        return reward

    def _modeling(self):
        sim_env = simpy.Environment()
        monitor = Monitor(self.record_events)

        jobs = []
        df_scenario_group = self.data.groupby(by=["Job_Name", "Job_Index", "Arrival_Date", "Due_Date"])

        for idx, group in df_scenario_group:
            job_name, job_index, arrival_date, due_date = idx

            operations = []
            for i, row in group.iterrows():
                options = np.array(row.iloc[6:].to_list())
                operation = Operation(row["Operation_Name"], row["Operation_Index"], options=options)
                operations.append(operation)

            job = Job(job_name, job_index, arrival_date, due_date, operations)
            jobs.append(job)

        jobs = sorted(jobs, key=lambda x: x.arrival_date)

        model = {}
        model["Source"] = Source(sim_env, "Source", jobs, model, monitor)
        for i, name in enumerate(self.data.columns[6:]):
            machine = Machine(sim_env, name, i, model, monitor, capacity=1)
            model[name] = machine
        model["Buffer"] = Buffer(sim_env, "Buffer", model, monitor)
        model["Sink"] = Sink(sim_env, "Sink", monitor)

        return sim_env, jobs, model, monitor