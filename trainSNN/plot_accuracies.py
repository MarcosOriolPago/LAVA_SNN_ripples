import os
import sys
import matplotlib.pyplot as plt


def split_train_test(data):
    train = []
    test = []
    for line in data.readlines():
        try:
            line = line.strip().split('    ')
            train.append(float(line[0]))
            test.append(float(line[1]))
        except:
            pass

    return train, test


trains = {}
tests = {}
parent = sys.argv[1]

for arch in os.listdir(parent):
    subparent = f'{parent}/{arch}/'
    if os.path.isdir(subparent):
        model_train_ys = {}
        model_test_ys = {}
        for model in os.listdir(subparent):
            if os.path.isdir(f'{subparent}/{model}'):
                acc_path = f'{subparent}{model}/accuracy.txt'
                data = open(acc_path, 'r')
                train, test = split_train_test(data)
                model_train_ys[f'{model.split("_")[1]}'] = train
                model_test_ys[f'{model.split("_")[1]}'] = test

        trains[f'{arch}'] = model_train_ys
        tests[f'{arch}'] = model_test_ys

del arch, model


# Define a custom style
plt.style.use('seaborn-v0_8-whitegrid')

for arch in trains:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    plt.subplots_adjust(hspace=0.5)
    for model in trains[arch]:
        axes[0].plot(trains[arch][model], label=model)
        axes[0].legend()
        axes[0].set_title('Train')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')

        axes[1].plot(tests[arch][model], label=model)
        axes[1].legend()
        axes[1].set_title('Test')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
    
    plt.suptitle(f'Architecture\n{arch}', fontweight='bold')
    plt.savefig(f'{parent}/{arch}/performances.svg', format='svg')