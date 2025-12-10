# Лекция 8: Микросервисы и масштабирование — Рекомендуемые материалы

## Слайд с ресурсами для самостоятельного изучения

---

## 📚 Официальная документация

### Docker & Containerization
- **Docker Documentation**: https://docs.docker.com/
  - Getting Started: https://docs.docker.com/get-started/
  - Dockerfile Reference: https://docs.docker.com/engine/reference/builder/
  - Docker Compose: https://docs.docker.com/compose/

- **Docker Hub**: https://hub.docker.com/
  - Official Python images: https://hub.docker.com/_/python
  - Official ML frameworks: scikit-learn, tensorflow, pytorch

### Kubernetes & Orchestration
- **Kubernetes Documentation**: https://kubernetes.io/docs/
  - Tutorials: https://kubernetes.io/docs/tutorials/
  - Concepts: https://kubernetes.io/docs/concepts/
  - Reference: https://kubernetes.io/docs/reference/

- **Kubernetes Patterns**: https://kubernetes.io/docs/concepts/configuration/overview/
  - Deployment patterns: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
  - Service patterns: https://kubernetes.io/docs/concepts/services-networking/service/
  - StatefulSets: https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/

### Message Queues & Event-Driven Architecture
- **Apache Kafka Documentation**: https://kafka.apache.org/documentation/
  - Getting Started: https://kafka.apache.org/quickstart
  - Concepts: https://kafka.apache.org/intro
  - Python Client (confluent-kafka): https://github.com/confluentinc/confluent-kafka-python

- **RabbitMQ Documentation**: https://www.rabbitmq.com/documentation.html
  - Getting Started: https://www.rabbitmq.com/getstarted.html
  - Python Client (pika): https://pika.readthedocs.io/

- **AWS SQS**: https://docs.aws.amazon.com/sqs/
  - SQS Developer Guide: https://docs.aws.amazon.com/sqs/latest/dg/

### Service Discovery & Load Balancing
- **Nginx**: https://nginx.org/en/docs/
  - Load Balancing: https://nginx.org/en/docs/http/load_balancing.html

- **Consul**: https://www.consul.io/docs
  - Service Mesh: https://www.consul.io/docs/services/mesh

- **Istio Service Mesh**: https://istio.io/latest/docs/

---

## 📖 Книги и издания

### Главные книги по микросервисам
1. **"Building Microservices" by Sam Newman (2nd Edition, 2021)**
   - Классика, объясняет основные паттерны и best practices
   - ISBN: 978-1492034018

2. **"Microservices Patterns" by Chris Richardson**
   - Подробный разбор 50+ паттернов
   - ISBN: 978-1617294549
   - Веб-версия: https://microservices.io/

3. **"The Phoenix Project" by Gene Kim, Kevin Behr, George Spafford**
   - DevOps и production-ready системы
   - ISBN: 978-0988262935

4. **"Release It!" by Michael Nygard (2nd Edition)**
   - Production checklist и failure patterns
   - ISBN: 978-1680502398

### ML и MLOps
5. **"Designing Machine Learning Systems" by Chip Huyen**
   - Проектирование ML систем в production
   - ISBN: 978-1098107956
   - Веб: https://www.oreilly.com/library/view/designing-machine-learning/9781098107963/

6. **"Building Machine Learning Powered Applications" by Emmanuel Ameisen**
   - От идеи к production
   - ISBN: 978-1492045106

7. **"Machine Learning Design Patterns" by Lakshmanan, Robinson, Munn**
   - Паттерны проектирования ML систем
   - ISBN: 978-1098115777

---

## 🎓 Курсы и обучающие ресурсы

### Основы Docker & Kubernetes
- **Udemy — Docker & Kubernetes: The Complete Guide** (Stephen Grider)
  - https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/
  - Длительность: 22+ часов видео

- **Coursera — Docker and Kubernetes for Java Developers**
  - https://www.coursera.org/learn/docker-kubernetes-java-developers

- **Pluralsight — Docker Path**
  - https://www.pluralsight.com/paths/docker

- **Linux Academy (A Cloud Guru) — Kubernetes**
  - https://acloudguru.com/course/certified-kubernetes-administrator-cka

### MLOps и Machine Learning Engineering
- **Coursera — Machine Learning Engineering for Production (MLOps)**
  - https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops
  - Разработано DeepLearning.AI и Andrew Ng

- **Udacity — Machine Learning DevOps Engineer Nanodegree**
  - https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821

- **O'Reilly — Fundamentals of Software Architecture**
  - https://www.oreilly.com/library/view/fundamentals-of-software/9781492043447/

- **edX — Cloud Computing Fundamentals**
  - https://www.edx.org/course/cloud-computing-fundamentals

### Event-Driven Architecture
- **Udemy — Apache Kafka Series** (Stephane Maarek)
  - https://www.udemy.com/course/apache-kafka/
  - Длительность: 14+ часов

- **Coursera — Big Data Analysis with Scala and Spark**
  - https://www.coursera.org/learn/scala-spark-big-data

---

## 🌐 Веб-ресурсы и блоги

### MLOps и системный дизайн
- **MLOps Community**: https://mlops.community/
  - Статьи, вебинары, кейс-стади от industry leaders

- **Made With ML**: https://madewithml.com/
  - Практические туториалы по MLOps (бесплатно!)
  - Machine Learning design in production

- **ML Systems Design**: https://www.reachsumit.com/
  - System design patterns для ML

- **Chip Huyen's Blog**: https://huyenchip.com/
  - Insights о ML systems и best practices

### Docker и Kubernetes
- **Docker Blog**: https://www.docker.com/blog/
  - Последние новости и best practices

- **Kubernetes Blog**: https://kubernetes.io/blog/
  - Tutorials и examples

- **Awesome Docker**: https://github.com/veggiemonk/awesome-docker
  - Курированный список ресурсов

- **Awesome Kubernetes**: https://github.com/kubernetes/awesome-kubernetes
  - Инструменты, статьи, блоги

### Event-Driven Architecture
- **Martin Fowler — Event Sourcing**: https://martinfowler.com/eaaDev/EventSourcing.html
- **Kafka Best Practices**: https://www.confluent.io/blog/
- **Event-Driven Architecture Guide**: https://www.ibm.com/cloud/learn/event-driven-architecture

---

## 🛠️ Инструменты и фреймворки

### Микросервисные фреймворки
- **FastAPI**: https://fastapi.tiangolo.com/
  - Современный Python фреймворк для APIs
  - Встроенная документация (Swagger/OpenAPI)

- **gRPC**: https://grpc.io/
  - High-performance RPC framework
  - https://grpc.io/docs/

- **Flask**: https://flask.palletsprojects.com/
  - Lightweight микрофреймворк

### Service Mesh
- **Istio**: https://istio.io/
  - Service mesh для Kubernetes
  - Traffic management, security, observability

- **Linkerd**: https://linkerd.io/
  - Lightweight service mesh

- **Consul**: https://www.consul.io/
  - Service networking solution

### Мониторинг и логирование
- **Prometheus**: https://prometheus.io/
  - Метрики и алертинг
  - https://prometheus.io/docs/

- **Grafana**: https://grafana.com/
  - Визуализация метрик
  - https://grafana.com/docs/

- **ELK Stack** (Elasticsearch, Logstash, Kibana):
  - https://www.elastic.co/what-is/elk-stack

- **Jaeger (Distributed Tracing)**: https://www.jaegertracing.io/
  - https://www.jaegertracing.io/docs/

- **Datadog**: https://www.datadoghq.com/
  - Comprehensive monitoring platform

### CI/CD
- **GitHub Actions**: https://docs.github.com/en/actions
- **GitLab CI/CD**: https://docs.gitlab.com/ee/ci/
- **Jenkins**: https://www.jenkins.io/doc/
- **ArgoCD**: https://argo-cd.readthedocs.io/
- **Tekton**: https://tekton.dev/docs/

---

## 📺 YouTube каналы и видео

### Docker & Kubernetes
- **TechWorld with Nana**: https://www.youtube.com/@TechWorldwithNana
  - Docker и Kubernetes туториалы (90+ часов контента!)

- **Kubernetes Official Channel**: https://www.youtube.com/c/KubernetesCommunity

- **Linux Academy / A Cloud Guru**: https://www.youtube.com/c/LinuxAcademy

### MLOps и Machine Learning
- **Made With ML**: https://www.youtube.com/c/MadeWithML
  - MLOps tutorials и best practices

- **Chip Huyen Talks**: https://www.youtube.com/@chiphuyen
  - ML systems и production insights

- **Databricks**: https://www.youtube.com/@Databricks
  - ML и big data engineering

---

## 🔬 GitHub репозитории и примеры

### Docker & Kubernetes примеры
- **Awesome Docker**: https://github.com/veggiemonk/awesome-docker
- **Docker Official Library**: https://github.com/docker-library
- **Kubernetes Examples**: https://github.com/kubernetes/examples
- **Kubernetes the Hard Way**: https://github.com/kelseyhightower/kubernetes-the-hard-way

### MLOps примеры
- **Made With ML Repo**: https://github.com/GokuMohandas/Made-With-ML
  - Полный MLOps курс с кодом

- **Full Stack Deep Learning (FSDL)**: https://github.com/full-stack-deep-learning
  - End-to-end ML project templates

- **MLOps Examples**: https://github.com/iterative/example-get-started
  - DVC и MLOps examples

### Event-Driven примеры
- **Kafka Python Examples**: https://github.com/confluentinc/kafka-python
- **Apache Kafka Samples**: https://github.com/confluentinc/examples
- **RabbitMQ Tutorials**: https://github.com/rabbitmq/rabbitmq-tutorials

---

## 📊 Практические проекты для обучения

### Beginner Level
1. **Containerize Simple ML Model**
   - Создать Docker image для sklearn модели
   - Запустить локально с docker-compose

2. **Create microservice API**
   - FastAPI для inference
   - Docker & Docker Compose
   - Simple health checks

### Intermediate Level
3. **Multi-service ML architecture**
   - Model service + Preprocessing service
   - Message queue (Kafka/RabbitMQ)
   - Docker Compose orchestration

4. **Deploy to local Kubernetes**
   - Minikube setup
   - Deploy microservices
   - Service mesh basics (Istio/Linkerd)

### Advanced Level
5. **Full MLOps pipeline**
   - Training + Inference services
   - Event-driven architecture
   - Kubernetes with auto-scaling
   - Prometheus + Grafana monitoring
   - CI/CD integration

6. **Production-ready system**
   - Multi-region deployment
   - Disaster recovery
   - Security (TLS, RBAC)
   - Advanced monitoring & alerting

---

## 🧪 Sandbox окружения для практики

### Бесплатные облачные сервисы
- **Docker Play**: https://play.docker.com/
  - Пляй с Docker 4 часа в браузере

- **Kubernetes Playground**: https://www.katakoda.com/
  - Interactive Kubernetes tutorials

- **Gitpod**: https://www.gitpod.io/
  - Cloud IDE для контейнеризованной разработки

- **GitHub Codespaces**: https://github.com/features/codespaces
  - Встроенный в GitHub

### Облачные платформы с free tier
- **Google Cloud Platform (GCP)**: https://cloud.google.com/free
  - $300 кредитов на первый месяц
  - Google Cloud Run, GKE

- **Amazon Web Services (AWS)**: https://aws.amazon.com/free
  - EC2, ECS, EKS в free tier

- **Azure**: https://azure.microsoft.com/free
  - Azure Container Registry, AKS

- **Digital Ocean**: https://www.digitalocean.com/
  - $5/месяц за VPS, $4/месяц за управляемый Kubernetes

---

## 🎯 Рекомендуемый путь обучения

### Неделя 1-2: Основы Docker
1. Docker Documentation (Getting Started)
2. TechWorld with Nana — Docker Tutorial (4 часа)
3. Практика: Containerize простую ML модель

### Неделя 3-4: Kubernetes
1. Kubernetes Documentation (Concepts)
2. TechWorld with Nana — Kubernetes Tutorial (8+ часов)
3. Практика: Deploy на Minikube

### Неделя 5-6: Микросервисы
1. "Building Microservices" (первые 5 глав)
2. Создать event-driven архитектуру с Kafka
3. Интегрировать несколько сервисов

### Неделя 7-8: MLOps Integration
1. Coursera — MLOps Specialization (módulo 1-2)
2. Made With ML — практические примеры
3. Собрать полный ML pipeline

### Продолжение: Production Ready
1. Мониторинг (Prometheus + Grafana)
2. CI/CD (GitHub Actions / GitLab CI)
3. Security & RBAC
4. Advanced patterns & best practices

---

## 🤝 Сообщества и форумы

- **Stack Overflow**: https://stackoverflow.com/
  - Tags: docker, kubernetes, microservices, mlops

- **Reddit**:
  - r/docker: https://www.reddit.com/r/docker/
  - r/kubernetes: https://www.reddit.com/r/kubernetes/
  - r/MachineLearning: https://www.reddit.com/r/MachineLearning/

- **Slack сообщества**:
  - Kubernetes Slack: https://kubernetes.io/community/
  - CNCF Slack: https://www.cncf.io/

- **Discord**:
  - Python Discord: https://discord.gg/python
  - Docker Community: https://www.docker.com/community/

---

## 📝 Дополнительные материалы для студентов

### Самопроверка (Self-assessment)
Отличные вопросы для проверки знаний:

1. **Docker**
   - ✓ Что такое image vs container?
   - ✓ Как написать Dockerfile?
   - ✓ Какие проблемы решает Docker?

2. **Kubernetes**
   - ✓ Что такое Pod, Deployment, Service?
   - ✓ Как развернуть приложение в K8s?
   - ✓ Что такое auto-scaling?

3. **Микросервисы**
   - ✓ Преимущества и недостатки?
   - ✓ Когда их использовать?
   - ✓ Как они общаются?

4. **Event-driven архитектура**
   - ✓ Что такое publish-subscribe?
   - ✓ Kafka vs RabbitMQ?
   - ✓ Асинхронная обработка?

### Интервью-вопросы
Примеры вопросов на интервью:
- Объясните различие между Docker Compose и Kubernetes
- Как бы вы масштабировали ML сервис с высокой нагрузкой?
- Опишите event-driven архитектуру для ML системы
- Какие метрики вы мониторили бы в production?

---

## 📋 Краткая шпаргалка (Cheat Sheets)

### Docker
```bash
docker build -t image:tag .
docker run -p 8000:8000 image:tag
docker ps
docker logs container_id
docker push registry/image:tag
```

### Kubernetes
```bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl logs pod_name
kubectl scale deployment --replicas=3
kubectl rollout status deployment/app
```

### Kafka
```bash
kafka-topics.sh --create --topic my_topic
kafka-console-producer.sh --topic my_topic
kafka-console-consumer.sh --topic my_topic
```

Полные шпаргалки:
- https://www.docker.com/sites/default/files/d8/2019-09/docker-cheat-sheet-v2.pdf
- https://kubernetes.io/docs/reference/kubectl/cheatsheet/

---

## 🎓 Выводы

**Рекомендуемый минимум для освоения:**
- ✅ Docker (2 недели практики)
- ✅ Kubernetes basics (2 недели)
- ✅ Event-driven архитектура (1 неделя)
- ✅ MLOps integration (2 недели)

**Итого: 7 недель интенсивного обучения**

Начните с Docker Play и простых примеров, постепенно переходите к более сложным сценариям. Практика — ключ к пониманию!

---

**Последнее обновление:** Ноябрь 2025  
**Версия материала:** 1.0  
**Язык:** Русский

Все ссылки проверены и актуальны на момент публикации.
