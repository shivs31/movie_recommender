"""
TODO 

write a program that checks that recommenders works as expected

we will use pytest
to install pytest in the terminal

+pip install pytest
/conda install pytest

TDD (Test Driven Development) cycle:

0. Make a Hypothesis:
          the units/program work
1. Write test that fails(to disprove hypothesis)
2. Change the code so that the hypothesis is re-estabilished
3. repeat 0-->2

"""
from recommenders import MOVIES, random_recommender

def test_movies_are_strings():
    for movie in MOVIES:
        assert isinstance(movie,str)

def test_for_two_movies():
    top2 = random_recommender(k=2)
    assert len(top2) == 2

def test_for_5_movies():
    top5 = random_recommender(k=4)
    assert len(top5) == 4

def test_for_retun_0_if_k_10_movies():
    top10 = random_recommender(k=10)
    assert len(top10) == 0
    