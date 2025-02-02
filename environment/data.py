import numpy as np
import pandas as pd

from utils.visualize import WIP_graph


class DataGenerator:
    def __init__(self, config):
        self.n_machines = config.n_machines
        self.n_jobs = config.n_jobs
        self.n_init_jobs = config.n_init_jobs
        self.n_options_min = config.n_options_min
        self.n_options_max = config.n_options_max
        self.n_operations_min = config.n_operations_min
        self.n_operations_max = config.n_operations_max
        self.proctime_min = config.proctime_min
        self.proctime_max = config.proctime_max
        self.iat_avg = config.iat_avg
        self.ddt = config.ddt

    def generate(self, file_path=None):
        columns = ["Job_Name", "Job_Index", "Arrival_Date", "Due_Date",
                   "Operation_Name", "Operation_Index", "Order", "Start_Date", "Finish_Date"]
        columns = columns + ["Machine %d" % i for i in range(self.n_machines)]

        temp = []
        offset = 0
        expected_finish_dates = []
        for i in range(self.n_jobs):
            job_name = "J-%d" % i
            job_index = i

            if i == 0:
                arrival_date = 0
            else:
                iat = int(np.random.geometric(1 / self.iat_avg))
                arrival_date += iat


            num_operations = np.random.randint(self.n_operations_min, self.n_operations_max + 1)
            proctime_sum = 0.0
            start_date = arrival_date
            for j in range(num_operations):
                operation_name = "O-%d%d" % (i, j)
                operation_index = offset + j
                order = j

                proctime = np.zeros(self.n_machines)
                num_options = np.random.randint(self.n_options_min, self.n_options_max + 1)
                options = np.random.choice(range(self.n_machines), num_options, replace=False)
                proctime_avg = np.random.randint(self.proctime_min, self.proctime_max + 1)
                proctime_sampled = [np.random.randint(np.ceil(0.8 * proctime_avg),
                                                      np.floor(1.2 * proctime_avg) + 1)
                                    for _ in range(num_options)]
                # proctime_sum += np.mean(proctime_sampled)
                proctime_sum += np.max(proctime_sampled)
                proctime[options] = proctime_sampled

                row = ([job_name, job_index, arrival_date, 0,
                        operation_name, operation_index, order, start_date, start_date + np.mean(proctime_sampled)]
                       + list(proctime))
                temp.append(row)

                start_date += np.mean(proctime_sampled)

            expected_finish_dates.append(start_date)

            for k in range(offset, offset + num_operations):
                temp[k][3] = int(proctime_sum * self.ddt)

            offset += num_operations

        df_scenario = pd.DataFrame(temp, columns=columns)

        start = np.random.randint(df_scenario["Arrival_Date"].min(), int(min(expected_finish_dates)))
        df_initial = df_scenario[(df_scenario["Start_Date"] <= start) & (df_scenario["Finish_Date"] >= start)]
        df_initial = df_initial.sort_values(by=["Start_Date"])
        df_initial = df_initial.reset_index(drop=True)

        occupied_machines = []
        for i, row in df_initial.iterrows():
            machine_list = row.iloc[9:]
            machine_list = machine_list[(machine_list != 0) & ~(machine_list.index.isin(occupied_machines))]

            if len(machine_list) == 0:
                machine = "Buffer"
            else:
                machine = machine_list.sample(n=1).index.to_numpy()[0]
                occupied_machines.append(machine)

            df_initial.loc[i, "Initial_Machine"] = machine

        if file_path is not None:
            writer = pd.ExcelWriter(file_path)
            df_scenario.to_excel(writer, sheet_name="scenario", index=False)
            df_initial.to_excel(writer, sheet_name="initial", index=False)
            writer.close()

        return df_scenario, df_initial


if __name__ == '__main__':
    import os
    import argparse

    def get_config():
        parser = argparse.ArgumentParser(description="FJSP")

        parser.add_argument("--n_jobs", type=int, default=10, help="number of jobs")
        parser.add_argument("--n_init_jobs", type=int, default=5, help="number of jobs")
        parser.add_argument("--n_machines", type=int, default=5, help="number of machines")
        parser.add_argument("--n_operations_min", type=int, default=4, help="minimum number of operations per job")
        parser.add_argument("--n_operations_max", type=int, default=6, help="maximum number of operations per job")
        parser.add_argument("--n_options_min", type=int, default=1, help="minimum number of available machines")
        parser.add_argument("--n_options_max", type=int, default=5, help="maximum number of available machines")
        parser.add_argument("--proctime_min", type=int, default=1, help="minimum processing time")
        parser.add_argument("--proctime_max", type=int, default=20, help="maximum processing time")
        parser.add_argument("--iat_avg", type=float, default=4, help="average inter-arrival time")
        parser.add_argument("--ddt", type=float, default=1.2, help="due date tardiness")

        return parser.parse_args()

    config = get_config()

    file_dir = "../input/validation/%d-%d/" % (config.n_jobs, config.n_machines)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    data_generator = DataGenerator(config)
    n_instance = 20
    for i in range(1, n_instance + 1):
        file_path = file_dir + "instance-{0}.xlsx".format(i)
        data_generator.generate(file_path=file_path)
        # flag = True
        # while flag:
        #     file_path = file_dir + "instance-{0}.xlsx".format(i)
        #     df_scenario, df_initial = data_generator.generate(file_path=file_path)
        #     max_wip = WIP_graph(df_scenario)
        #     if config.n_machines * 0.8 <= max_wip <= config.n_machines * 1.2:
        #         flag = False