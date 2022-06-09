# XAI-Metric-Evaluator

Данное программное обеспечение выполняет вычисление метрик оценки методов объяснимого искусственного интеллекта SHAP, LIME с использованием синтетических данных.

### Установка и запуск
1. Необходимо иметь установленный Python 3.9
2. Создать виртуальное окружение (venv) и активировать его
3. Установить зависимости командой `pip install requirements.txt`
4. Пример запуска программы с конфигурацией experiment.jsonc - скрипт run.sh
5. Для изучения параметров командной строки для программы (main.py) запустить `main.py --help`

### Добавление имплементации методов XAI
1. В папке xai_methods создать файл для класса метода XAI
2. Реализовать класс метода XAI согласно данному интерфейсу:
``` Python
class Method:
    def __init__(self, f, x, **kwargs):
      # f - модель машинного обучения
      # x - датасет
      # kwargs - дополнительные параметры (могут быть указаны в файле конфигурации)
    def explain(self, x):
      # x - набор данных
      # возвращает вектор весов признаков w
```
3. Импортировать класс в файле `__init__.py` и добавить в объект `available_methods`.
``` Python
available_methods = {
    "shap": ShapXAI,
    "kernelshap": KernelShap,
    "lime": LimeXAI,
    # "your_method": YourMethodClass
}
```

### Аргументы командной строки

  --mode {classification,regression}
                        Classification or regression?
                        
  --dataset DATASET     Name of the dataset to train on
  
  --model MODEL         Algorithm to use for training
  
  --explainer EXPLAINER
                        Explainer to use
                        
  --metric METRIC       Metric to evaluate the explanation
  
  --data-kwargs DATA_KWARGS
                        Custom data args needed to generate the dataset.\n Default = '{}'
                        
  --data-kwargs-json DATA_KWARGS_JSON
                        Path to json file containing custom data args.
                        
  --model-kwargs MODEL_KWARGS
                        Custom data args needed to generate the dataset.\n Default = '{}'
                        
  --model-kwargs-json MODEL_KWARGS_JSON
                        Path to json file containing custom data args.
                        
  --seed SEED           Setting a seed to make everything deterministic.
  
  --experiment          Run multiple experiments using an experiment config file.
  
  --rho RHO             Control the rho of an experiment.
  
  --rhos RHOS [RHOS ...]
                        Control the rhos of a mixture experiment.
                        
  --experiment-json EXPERIMENT_JSON
  
  --no-logs             whether to save results or not. You can use this avoid overriding your result files while testing.
  
  --results-dir RESULTS_DIR
                        Path to save results in csv files.
