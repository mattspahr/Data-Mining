from sklearn.linear_model import LogisticRegression
import numpy as np
import copy


def main():
    # I found the new matrix based on the 10 users and 8 movies listed on the HW7.pdf
    movies = ["Captain America", "Non-Stop", "The Wolf of Wall Street", "Frozen",
              "300 Rise of an Empire", "THOR", "Fate of the Furious", "The Boss Baby"]

    ratings = [[1, 1, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
            [1, 0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 1, 1]]

    pred = copy.deepcopy(ratings)
    n_users = len(pred)
    n_movies = len(pred[0])
    print "Probabilities: "
    for u in range(n_users):
        for m in range(n_movies):
            if pred[u][m] == 0:  # If it equals 1, this movie will not be recommended since the user has watched the movie already.
                Y = []
                X = []
                for i in range(u) + range(u + 1, n_users):
                    X.append(ratings[i][:m] + ratings[i][m + 1:])
                    Y.append(ratings[i][m])
                lr = LogisticRegression()
                lr.fit(X, Y)
                val = lr.predict_proba([ratings[u][:m] + ratings[u][m + 1:]])
                pred[u][m] = round(val[0][1], 2)
    for vals in pred:
        print vals

    print "\nRecommendation System: "

    maxval = 0

    for vals in pred:
        max = 0
        maxI = 0
        for val in vals:
            if val >= max and val != 1:
                max = val
                maxI = vals.index(val)
        print "Recommendation for User {} is: {}".format(pred.index(vals) + 1, movies[maxI])

if __name__ == '__main__':
    main()