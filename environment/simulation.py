import pandas as pd


class Operation:
    def __init__(self, name, id, options):
        self.name = name
        self.id = id
        self.options = options

        self.progress = 0.0
        self.start_time = None
        self.finish_time = None
        self.current_machine = None

    def get_processing_time(self, machine_id):
        return self.options[machine_id]


class Job:
    def __init__(self, name, id, arrival_date, due_date, operations):
        self.name = name
        self.id = id
        self.arrival_date = arrival_date
        self.due_date = due_date
        self.operations = operations

        self.step = 0
        self.waiting_start = 0
        self.current_machine = None
        self.actual_arrival_date = 0

    def get_current_operation(self):
        if len(self.operations) == self.step:
            operation = None
        else:
            operation = self.operations[self.step]
        return operation


class Source:
    def __init__(self, env, name, jobs, model, monitor):
        self.env = env
        self.name = name
        self.jobs = jobs
        self.model = model
        self.monitor = monitor

        self.calling_event = {}
        self.action = env.process(self.run())

        self.sent = 0

    def run(self):
        for job in self.jobs:
            self.monitor.jobs_before_arrival[job.id] = job
            for i, operation in enumerate(job.operations):
                self.monitor.operations_unscheduled[operation.id] = operation

        while True:
            job = self.jobs[self.sent]

            IAT = job.arrival_date - self.env.now
            if IAT > 0:
                yield self.env.timeout(IAT)
            self.env.process(self.arrive(job))

            self.sent += 1

            if len(self.jobs) == self.sent:
                break

    def arrive(self, job):
        del self.monitor.jobs_before_arrival[job.id]
        self.monitor.jobs_in_process[job.id] = job

        job.actual_arrival_date = self.env.now

        self.monitor.put_queue(job)
        self.calling_event[job.name] = self.env.event()
        next_machine = yield self.calling_event[job.name]

        self.model[next_machine].put(job)
        del self.calling_event[job.name]

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name, event="Job Arrived")


class Machine:
    def __init__(self, env, name, id, model, monitor, capacity=1):
        self.env = env
        self.name = name
        self.id = id
        self.model = model
        self.monitor = monitor
        self.capacity = capacity

        self.calling_event = {}
        self.operations_in_working = {}
        self.processes = {}

        self.working_time = 0.0
        self.completion_time = 0.0

    def put(self, job):
        operation = job.get_current_operation()
        job.current_machine = self.name
        operation.current_machine = self.name
        self.processes[operation.id] = self.env.process(self.working(job))
        self.operations_in_working[operation.id] = operation

        del self.monitor.operations_unscheduled[operation.id]
        self.monitor.operations_in_machine[operation.id] = operation

    def working(self, job):
        operation = job.get_current_operation()

        processing_time = operation.get_processing_time(self.id)
        operation.start_time = self.env.now

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name,
                                operation=operation.name, event="Working Started")

        self.completion_time = self.env.now + processing_time
        yield self.env.timeout(processing_time)
        self.working_time += processing_time

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name,
                                operation=operation.name, event="Working Finished")
        job.step += 1
        operation.finish_time = self.env.now
        self.monitor.operations_done[operation.id] = operation

        del self.processes[operation.id]
        del self.operations_in_working[operation.id]
        del self.monitor.operations_in_machine[operation.id]

        if job.step == len(job.operations):
            self.model["Sink"].put(job)
            if len(self.monitor.operations_in_buffer) > 0:
                self.monitor.scheduling = True
        else:
            self.monitor.put_queue(job)
            self.calling_event[job.name] = self.env.event()
            next_machine = yield self.calling_event[job.name]

            del self.calling_event[job.name]
            self.model[next_machine].put(job)

    def check_status(self):
        idle = False
        available_time = self.env.now
        if len(self.operations_in_working) == 0:
            idle = True
        else:
            for id, operation in self.operations_in_working.items():
                proc_time = operation.get_processing_time(machine_id=self.id)
                temp = operation.start_time + proc_time
                if temp > available_time:
                    available_time = temp
        return idle, available_time


class Buffer:
    def __init__(self, env, name, model, monitor, capacity=float('inf')):
        self.env = env
        self.name = name
        self.model = model
        self.monitor = monitor
        self.capacity = capacity

        self.calling_event = {}
        self.operations_in_waiting = {}
        self.processes = {}

    def put(self, job):
        operation = job.get_current_operation()
        job.current_machine = self.name
        operation.current_machine = self.name
        self.processes[operation.id] = self.env.process(self.waiting(job))
        self.operations_in_waiting[operation.id] = operation
        self.monitor.operations_in_buffer[operation.id] = operation

    def waiting(self, job):
        job.waiting_start = self.env.now
        operation = job.get_current_operation()

        self.monitor.put_queue(job, from_buffer=True)

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name,
                                operation=operation.name, event="Waiting Started")

        self.calling_event[job.name] = self.env.event()
        next_machine = yield self.calling_event[job.name]

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name,
                                operation=operation.name, event="Waiting Finished")

        del self.calling_event[job.name]
        del self.processes[operation.id]
        del self.operations_in_waiting[operation.id]
        del self.monitor.operations_in_buffer[operation.id]

        self.model[next_machine].put(job)


class Sink:
    def __init__(self, env, name, monitor):
        self.env = env
        self.name = name
        self.monitor = monitor

        self.event = env.event()
        self.jobs = []
        self.completion_time = 0.0
        self.total_tardiness = 0.0

    def put(self, job):
        self.completion_time = self.env.now
        self.total_tardiness += max(self.env.now - job.due_date, 0)

        self.jobs.append(job)
        self.monitor.jobs_completed[job.id] = job
        del self.monitor.jobs_in_process[job.id]

        if self.monitor.record_events:
            self.monitor.record(self.env.now, location=self.name, job=job.name, event="Job Completed")


class Monitor:
    def __init__(self, record_events=True):
        self.record_events = record_events

        self.jobs_in_queue = {}
        self.scheduling = False

        self.operations_unscheduled = {}
        self.operations_in_machine = {}
        self.operations_in_buffer = {}
        self.operations_done = {}

        # self.jobs_all = {}
        self.jobs_before_arrival = {}
        self.jobs_in_process = {}
        self.jobs_completed = {}

        self.time = []
        self.location = []
        self.job = []
        self.operation = []
        self.event = []
        self.info = []

    def put_queue(self, job, from_buffer=False):
        if not self.scheduling and not from_buffer:
            self.scheduling = True
        self.jobs_in_queue[job.id] = job

    def remove_queue(self, job_id):
        job = self.jobs_in_queue[job_id]
        del self.jobs_in_queue[job_id]

        operations_for_decision = []
        for temp in self.jobs_in_queue.values():
            operation = temp.get_current_operation()
            if not operation.id in self.operations_in_buffer.keys():
                operations_for_decision.append(operation.id)

        if len(operations_for_decision) == 0:
            self.scheduling = False

        return job

    def record(self, time, location=None, job=None, operation=None, event=None, info=None):
        self.time.append(time)
        self.location.append(location)
        self.job.append(job)
        self.operation.append(operation)
        self.event.append(event)

    def get_logs(self, file_path=None):
        df_log = pd.DataFrame(columns=['Time', 'Location', 'Job', 'Operation', 'Event'])
        df_log['Time'] = self.time
        df_log['Location'] = self.location
        df_log['Job'] = self.job
        df_log['Operation'] = self.operation
        df_log['Event'] = self.event

        if file_path is not None:
            df_log.to_excel(file_path, index=False)

        return df_log