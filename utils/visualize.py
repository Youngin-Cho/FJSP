import numpy as np
import matplotlib.pyplot as plt


def WIP_graph(data, graph=False, filepath=None):
    start = data["Arrival_Date"].min()
    finish = data["Due_Date"].max()
    timeline = np.arange(start, finish + 1)
    wip = np.zeros(int(finish - start) + 1)

    df_group = data.groupby(by=["Job_Name", "Job_Index", "Arrival_Date", "Due_Date"])
    for idx, df_temp in df_group:
        job_name, job_index, arrival_date, due_date = idx

        start = arrival_date
        for i, row in df_temp.iterrows():
            proctimes = row.iloc[6:].to_numpy()
            mean_proctime = int(np.mean(proctimes[proctimes.nonzero()]))
            wip[start:start + mean_proctime] += 1
            start += mean_proctime

    if graph:
        fig, ax = plt.subplots(1, figsize=(16, 6))
        ax.plot(timeline, wip)
        plt.show()
        if filepath is not None:
            plt.savefig(filepath)
        plt.close()

    return np.max(wip)


if __name__ == "__main__":
    import pandas as pd
    data_path = "../input/validation/15-5/instance-11.xlsx"
    data = pd.read_excel(data_path)
    wip = WIP_graph(data, graph=True)