# itmo_reliableml
***
## Decription
Проект для прохождения курса ML System Design в рамках магистратуры ИТМО "Инженерия машинного обучения" <br>
### Тема проекта
Здесь будет тема проекта, когда придумаем
### Состав участников:
- Виталий Ахмадиев [@VitalyAkhmadiev](https://t.me/VitalyAkhmadiev)
- Дмитрий Паршин [@The_Illusive_Man_2000](https://t.me/The_Illusive_Man_2000)
- Павел Сауков [@a12c4](https://t.me/a12c4)

***
## Getting Started
Установку и запуск сервиса можно осуществить несколькими способами:
1. С помощью docker-контейнера
2. Клонировав git-репозиторий

## Installation with docker-container
### Prerequisites
Для запуска контейнера вам понадобится установить [docker](https://docs.docker.com/engine/install/ubuntu/):
```commandline
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
Для проверки, что установка прошла успешно запустить следующую команду:
```commandline
Verify that the Docker Engine installation is successful by running the hello-world image
```
### Install and Run
Из проекта достаточно скачать 2 файла
- docker-compose.yaml
- Dockerfile
Запустить следующую команду из директории, в которой находятся оба файла:
```commandline
docker-compose -f docker-compose.yaml up --build --detach
```

## Usage
После запуска контейнера сервис будет доступен по адресу: `http:\\localhost:8000`
Доступ к документации можно получить, обратившись: `http:\\localhost:8000\docs`

## Project status
Pet-проект
