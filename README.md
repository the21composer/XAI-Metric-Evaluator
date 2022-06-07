# XAI-Metric-Evaluator

Данное программное обеспечение выполняет вычисление метрик оценки методов объяснимого искусственного интеллекта SHAP, LIME с использованием синтетических данных.

### Установка и запуск
1. Необходимо иметь установленный Python 3.9
2. Создать виртуальное окружение (venv) и активировать его
3. Установить зависимости командой `pip install requirements.txt`
4. Пример запуска программы с конфигурацией experiment.jsonc - скрипт run.sh
5. Для изучения параметров командной строки для программы (main.py) запустить `main.py --help`

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
