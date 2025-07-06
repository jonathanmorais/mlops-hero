# run.py - ARQUIVO PRINCIPAL
from src.pipeline import executar_pipeline_completo

# UMA LINHA FAZ TUDO!
if __name__ == "__main__":
    modelo = executar_pipeline_completo("data/cs-training.csv")
    print("✅ Pronto!")