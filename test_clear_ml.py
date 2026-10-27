'''This code tests that ClearML works

run in the terminal

python test_clear_ml.py

'''

from clearml import Task

task = Task.init(project_name="PlantDisease", task_name="ClearML test")
logger = task.get_logger()

for i in range(5):
    logger.report_scalar("test", "value", value=i*2, iteration=i)

task.close()

