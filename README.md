# Python vs C++ for ML

L'un des plus grands obstacles à la démocratisation de l'apprentissage automatique est la performance. La plupart des algorithmes ML prennent beaucoup de temps à s'exécuter et sont coûteux en ressources.
Un meilleur matériel est la solution actuelle, mais cela coûte cher.

Cependant d'un point de vue logiciel, le langage de programmation le plus utilisé est Python. Même s'il est connu pour être l'un des langages de programmation les plus lents.

![Python vs C++ Benchmark](https://user-images.githubusercontent.com/71285263/191569304-bec1749f-ac1d-4773-86bf-576ab7730d29.png)

Cette image illustre des benchmarks effectués sur la même machine avec un code similaire en Python et en C++.

Il est donc possible de se demander pourquoi C++ n’est pas utilisé plus fréquemment.

Ce projet vise à explorer cette question. J'ai codé un algorithme ML simple dans les deux langages, K-nearest neighbour.

En Python, j'ai utilisé des librairies tierces mais je me suis abstenu de toute librairie tierce en C++ afin de refléter la réalité de l'utilisation de Python dans le contexte du ML.

## Traitement des données

J'ai utilisé le [jeu de données MNIST](https://en.wikipedia.org/wiki/MNIST_database).

### Python

Avec l’aide pandas et numpy, les données en Python sont relativement plus simples à lire. Ici, j’ai enregistré ces dernières dans des fichier `.npy` avant de les charger de mémoire.

### C++

Voici le [jeu de données](http://yann.lecun.com/exdb/mnist/) que j'ai utilisé en C++ car je ne pouvais pas utiliser les fichiers `.npy`. 

Les données étant en format binaire, leur lecture est simple en C++. Malgré cela, j’ai dû parse les données et ainsi créer un moyen de les gérer afin de pouvoir les déplacer lors des calculs.

Pour cela, j'ai créé la librairie [dataHandler](https://github.com/aryamaan3/cpp-vs-python-ml/tree/main/dataHandler).
La librairie dataHandler fournit deux choses :
- Un moyen de charger des données
- Une structure de données

Les données peuvent être chargées avec la classe DataHandler et la classe Data fournit une structure.

Cela a nécessité beaucoup plus de travail qu'en Python. Mais nous sommes prêts à tout pour gagner en performance.

## KNN

### Python

Encore une fois, l'implémentation de KNN en Python à l'aide de librairies comme numpy a été assez simple.

Ma fonction de prédiction calcule la distance euclidienne entre l'image à prédire et chaque image du jeu de données. Ensuite, les distances sont triés et le k le plus proche est renvoyé.

La distance euclidienne est calculée à l'aide de [numpy.linalg](https://numpy.org/doc/stable/reference/routines.linalg.html). Ce dernier fournit des implémentations de bas niveau de l'algèbre linéaire en Python. Par conséquent, il utilise des langages comme C pour accélérer le temps d'exécution.

### C++

J'ai créé une classe [KNN](https://github.com/aryamaan3/cpp-vs-python-ml/blob/main/knn/include/knn.hpp) qui utilise la classe [Data](https://github.com/aryamaan3/cpp-vs-python-ml/blob/main/dataHandler/include/data.hpp) de la librairie [dataHandler](https://github.com/aryamaan3/cpp-vs-python-ml/tree/main/dataHandler). Le constructeur prend k, les données d'apprentissage et les données de test.

Comme en Python, la fonction de prédiction calcule la distance euclidienne entre l'image à prédire et chaque image du jeu de données puis un tri pour renvoyer le k le plus proche. Tout cela est fait uniquement grâce à la librairie standard.

## Build

### Python

Comme Python est un langage interprété, aucune construction/compilation n'est nécessaire. Le code peut être exécuté directement liant ainsi automatiquement les librairies.

### C++

Cependant, en C++, le code doit être compilé manuellement avant l'exécution.
Cela devient fastidieux lorsqu’il s’agit de plusieurs librairies car elles doivent toutes être compilées avant de les exécuter.

J'ai utilisé [Makefile](https://opensource.com/article/18/8/what-how-makefile) pour automatiser le processus de compilation et ajouté un [wrapper bash](https://github.com/aryamaan3/cpp-vs-python-ml/blob/main/build_cpp.sh) qui :
- exécute le [Makefile](https://github.com/aryamaan3/cpp-vs-python-ml/blob/main/dataHandler/Makefile) de [ dataHandler](https://github.com/aryamaan3/cpp-vs-python-ml/tree/main/dataHandler)
- exécute ensuite le [Makefile](https://github.com/aryamaan3/cpp-vs-python-ml/blob/main/knn/Makefile) de [KNN](https://github.com/aryamaan3/cpp-vs-python-ml/tree/main/knn) qui dépend de [dataHandler](https://github.com/aryamaan3/cpp-vs-python-ml/tree/main/dataHandler)
- exécute enfin le programme.

## Résultats

Les deux programmes ont été exécutés sur la même machine. Le C++ est compilé avec l'optimisateur O3.

| Nb | k | Python | C++ | Diff(%) |
| --- | --- | --- | --- | --- |
| 10 | 1 | 3,0 | 2,4 | 25 |
| 10 | 2 | 3,2 | 2,7 | 18,5 |
| 10 | 3 | 3,4 | 2,8 | 21,4 |
| 10 | 4 | 2,9 | 2,4 | 20,8 |
| 100 | 1 | 29,8 | 25,4 | 17,3 |
| 100 | 2 | 28,4 | 24,5 | 15,9 |
| 100 | 3 | 28,0 | 24,2 | 15,7 |
| 100 | 4 | 30,0 | 23,7 | 26,6 |
| 1000 | 1 | 288 | 246 | 17,1 |
| 1000 | 2 | 292 | 241 | 21,2 |
| 1000 | 3 | 289 | 237 | 21,9 |
| 1000 | 4 | 291 | 234 | 24,4 |

On remarque que C++ surpasse Python de 20 % en moyenne.

## Conclusion

Les résultats montrent que C++ est 20 % plus rapide que Python sur ma machine. Cependant, il m'a fallu beaucoup plus de temps et d'efforts pour coder le programme en C++.

Alors, tout le monde devraient arrêter d'utiliser Python pour le ML ? Bien sûr que non. 

Concernant la démocratisation du ML, Python reste tout de même un meilleur choix grâce à sa simplicité. Ainsi, cela permet à un plus grand nombre de personne de l’utiliser pour le ML. 

De plus, la plupart des librairies Python utilisent des langages de bas niveau. Par exemple, 62 % du code de [Tensorflow](https://github.com/tensorflow/tensorflow), l'un des frameworks de deep learning les plus populaires, est écrit en C++.

Enfin, pour un algorithme simple comme KNN les avantages de Python l'emportent sur ceux de C++. Il serait cependant intéressant de comparer leurs performances sur des algorithmes de Deep Learning plus compliqués.

## Reproduction Local

Ouvrir un terminal dans la racine du projet

### Python

```bash
$ cd py_knn
```

```bash
$ python knn.py
```

### C++

```bash
$ source build_cpp.sh
```

Ou suivre les commandes dans [build_cpp.sh](https://github.com/aryamaan3/cpp-vs-python-ml/blob/main/build_cpp.sh) si vous n'êtes pas en bash

## Références

https://github.com/arinaschwartz/KNN-MNIST \
https://www.youtube.com/playlist?list=PL79n_WS-sPHKklEvOLiM1K94oJBsGnz71
