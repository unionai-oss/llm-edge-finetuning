from unionai.remote import UnionRemote
from flytekit.types.directory import FlyteDirectory


def get_recent_execution_id(workflow_name, remote):
    recent_executions = remote.recent_executions()
    executions = [e for e in recent_executions if e.spec.launch_plan.name == workflow_name]
    return executions[0].id.name


remote = UnionRemote()
workflow_name = "workflows.train_workflow"
execution_id = "f46d40e655d4e42f1b68"
execution = remote.fetch_execution(name=execution_id)
execution = remote.sync(execution, sync_nodes=True)
with remote.remote_context():
    model_directory: FlyteDirectory = execution.outputs["o0"]
    model_directory.path = "./model"
    model_directory.download()
