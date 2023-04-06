from sacred import Experiment

ex = Experiment("hello_config")


@ex.config
def my_config():
    recipient = "world"
    test = 1
    message = "Hello %s!" % recipient


@ex.config
def my_config_new(recipient: str):
    message = f"Come on {recipient}!"


@ex.automain
def my_main(message):
    print(message)
