import cProfile
import io
import numpy as np
import os
from pstats import SortKey
import pstats

from NeuralNetwork import NeuralNetwork
from Trainer import Trainer
from scripts.test import load_and_preprocess_mnist
"""
    Все комментарии и логи в этом файле на русском.
"""

def setup_profiling_environment():
    """
    Подготовка данных для профилирования.
    """
    print("Подготовка окружения для профилирования...")
    
    train_path = "MNIST_CSV/mnist_train.csv"
    test_path = "MNIST_CSV/mnist_test.csv"
    

    aug_params = {
        'phi': 10, 'scale': 0.1, 'trans_x': 2, 'trans_y': 2, 'noise_std': 0.01
    }
    X_train, y_train, X_test, y_test = load_and_preprocess_mnist(
        train_path, test_path, 
        use_full_dataset=True, 
        augmentation_config=aug_params
    )
    

    model = NeuralNetwork(
        input_size=784,
        hidden_layers=[128, 64],
        output_size=10,
        learning_rate=0.01,
        activation='relu',
        cost_function='cross_entropy'
    )
    
    return model, X_train, y_train

def run_training_session():
    """Функция-обертка для запуска тренировки, которую мы будем профилировать"""
    model, X, y = setup_profiling_environment()
    
    print("Запуск цикла...")
    trainer = Trainer(model, verbose=False) # verbose=False чтобы не засорять вывод
    
    trainer.train(
        X, y, 
        epochs=10,
        batch_size=32
    )

def profile_code():
    """Основная функция запуска профилировщика"""
    pr = cProfile.Profile()
    pr.enable()
    
    run_training_session()
    
    pr.disable()
    
    s = io.StringIO()
    
    sortby_cum = SortKey.CUMULATIVE
    ps_cum = pstats.Stats(pr, stream=s).sort_stats(sortby_cum)
    
    print("\n" + "="*60)
    print("ОТЧЕТ ПРОФИЛИРОВАНИЯ (Топ 20 по кумулятивному времени)")
    print("="*60)
    ps_cum.print_stats(20)
    
    print(s.getvalue())
    s.truncate(0) 
    s.seek(0)
    
    sortby_time = SortKey.TIME
    ps_time = pstats.Stats(pr, stream=s).sort_stats(sortby_time)
    
    print("\n" + "="*60)
    print("ОТЧЕТ ПРОФИЛИРОВАНИЯ (Топ 20 по чистому времени выполнения)")
    print("="*60)
    ps_time.print_stats(20)
    
    print(s.getvalue())
    
    pr.dump_stats('profile_stats.prof')
    print("\nПолная статистика сохранена в файл 'profile_stats.prof'")
    print("Вы можете визуализировать её командой: snakeviz profile_stats.prof")

if __name__ == "__main__":
    profile_code()
