# Antifrod-sys
Проект по предсказанию утечки клиентов по истории их транзакций. 

# Описание
В датасете представлены 4 характеристики:<br/>
user_id - индентификатор пользователя<br/>
date - дата и время транзакции<br/>
amount - сумма и тип транзакции<br/>
churn - ушел ли пользователь в бинарном виде - 0/1

# Мои результаты обучения

## График потерь (Loss)
![Loss vs Epoch](interim_results/train_loss.png)

## График точности (Accuracy)
![Accuracy vs Epoch](interim_results/val_roc_auc.png)


