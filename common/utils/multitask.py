class MultitaskConfig:
    """
    Wraps metadata for multitask training configuration. Each task configuration can be registered here and then values of fields can be retrieved for
    all of the registered tasks.
    """

    def __init__(self):
        self.tasks_config = {}

    def add_task_metadata(self, task_name, multitask_config):
        """
        Adds the task configuration.
        :param task_name: task name.
        :param multitask_config: dictionary with the configuration for the task.
        """
        self.tasks_config[task_name] = multitask_config

    def get_by_task_values(self, field_name):
        """
        Gets a dictionary of by task values, where the keys are task names and the values are the values of the given field name for each task.
        :param field_name: name of a field in the task's metadata.
        :return: dictionary of by task field values.
        """
        return {task_name: config[field_name] for task_name, config in self.tasks_config.items()}
