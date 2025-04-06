from sarcasm.data_handler import SarcasmDataHandler
from sarcasm.model.sarcasm_classifier import SarcasmClassifierBart


def main():
    model = SarcasmClassifierBart()
    data_handler = SarcasmDataHandler()

    # model.train(data_handler.dataset["train"])
    # model.save_model()
    metric = model.evaluate(data_handler.dataset["test"])
    print(metric)


if __name__ == "__main__":
    main()
