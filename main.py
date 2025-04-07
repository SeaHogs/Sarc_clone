from sarcasm.data_handler import SarcasmDataHandler, SarcasmGptDataHandler
from sarcasm.model.sarcasm_classifier import SarcasmClassifierBart
from sarcasm.model.sarcasm_remover import SarcasmRemoverBart


def main_classifier():
    model = SarcasmClassifierBart()
    data_handler = SarcasmDataHandler()

    model.train(data_handler.dataset["train"])
    model.save_model()
    metric = model.evaluate(data_handler.dataset["test"], metric="accuracy")
    print(metric)


def main_remover():
    model = SarcasmRemoverBart()
    data_handler = SarcasmGptDataHandler()

    model.train(data_handler.dataset["train"])
    model.save_model()
    print(
        model.predict(
            "onion social cracks down on sexual harassment by banning all women from platform"
        )
    )
    metric = model.evaluate(data_handler.dataset["test"])
    print(metric)


if __name__ == "__main__":
    main_classifier()
    # main_remover()
